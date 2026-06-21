在之前的流程中，我们一直默认 vLLM 已经完成了模型的加载。在本节课中，我们将深入学习 vLLM 关于模型加载的三个知识点，分别是：
1. 实例化模型结构；
2. 如何加载模型算子的权重、对权重的后处理以及融合算子的加载过程；
3. 模型结构的注册机制。
总体来看，模型的加载流程如下：
Executor → Worker.load_model() → GPUModelRunner.load_model() → BaseModelLoader.load_model()。
其中，真正负责模型查找与权重加载的核心逻辑，是 GPUModelRunner.load_model() 内部调用的 BaseModelLoader.load_model()，后文将沿着这条主线，依次分析BaseModelLoader.load_model()中模型类解析、权重加载和加载后处理这三个关键阶段，它们分别是：  
1. initialize_model()：完成模型类解析与实例化。该步骤会先调用 ModelRegistry.resolve_model_cls 从注册表中解析模型类，再根据 checkpoint 的 config.json 中的 architectures 字段找到对应的 vLLM 模型实现。例如，"LlamaForCausalLM" 会映射到 llama.py 中的 LlamaForCausalLM。随后完成模型实例化：  
model = model_class(vllm_config=vllm_config, prefix=prefix)
2. load_weights()：将 checkpoint 权重加载到模型参数中，并处理 fused 参数映射与并行分片。在这一阶段，vLLM 的模型级 load_weights 会接收来自ModelLoader的 (name, tensor) 迭代器，并逐参数完成加载。对于 qkv_proj、gate_up_proj 等 fused 参数，vLLM 会先将分开的权重映射到融合参数，再按对应分片完成加载；张量并行下的权重切分也在这一步完成
3. process_weights_after_loading()：执行加载后的权重后处理。在权重加载完成后，vLLM 会进一步执行后处理逻辑，包括量化权重重打包、融合算子初始化，以及 Attention 相关权重的后处理。
暂时无法在飞书文档外展示此内容
在前面的初始化流程中，用户传入的模型名称、模型路径、并行方式、数据类型等参数，会先被整理为 vLLM 内部的配置对象（包括self.vllm_config和self.model_config）。其中，与模型本身直接相关的部分会被封装为 model_config，并保存在ModelRunner实例的self.model_config中。
当执行到GPUModelRunner.load_model()时，该配置对象会继续向下传递，最终作为参数传入，供后续的模型类解析、权重加载以及加载后处理阶段使用。
暂时无法在飞书文档外展示此内容
model_loader.load_model(vllm_config=self.vllm_config, model_config=self.model_config)
在模型加载流程中，需要传入的参数不仅包括 vLLM 的通用配置，更关键的是与具体模型相关的 self.model_config。接下来，我们将对这个配置对象self.model_config进行 dump，并结合输出内容，重点关注其中两个关键字段：
- architecture = "Qwen3ForCausalLM"：表示当前模型的结构类型，后续 vLLM 会据此在模型注册表中解析出对应的模型类。
- model = "/home/test_fss/code/vllm_learn/Qwen3-1.7B"：表示待加载模型的实际路径，后续会从该目录中查找模型权重文件
architecture = 'Qwen3ForCausalLM'
model = '/home/test_fss/code/vllm_learn/Qwen3-1.7B

另外还有关于模型并行的信息，以下两条表示4路TP，模型权重按此切分
parallel_config.tensor_parallel_size=4
parallel_config.world_size=4
当前 worker 是第几个 TP rank，取值为（0,3）
parallel_config.rank=1
self.model_config中一个字段model用于指定当前模型的架构，另一个字段architecture用于指定模型的位置；该位置既可以是本地路径，也可以是远程地址。除这两个关键字段外，其他与模型加载相关的重要配置也统一保存在 model_config 中，例如分词器 tokenizer、模型数据类型 dtype、量化方式 quantization，以及最大上下文长度 max_model_len 等。
在拿到 model_config 之后，ModelRunner.load_model() 还需要根据 load_config 选择具体的模型加载器。这里会调用 get_model_loader(self.load_config)，根据当前的配置从预先注册好的加载器映射表中取出对应的 ModelLoader 实例。这样做的好处是，ModelRunner 本身不需要关心具体的权重文件如何读取，而是将这部分逻辑统一交给BaseModelLoader，一般是DefaultModelLoader。
暂时无法在飞书文档外展示此内容
- get_model_loader(self.load_config)：根据配置信息从注册表中选择对应的模型加载器实例，ModelLoader 封装了从不同来源（本地 / HF Hub / 云端）读取不同格式（safetensors / GGUF / 预分片等）的权重文件并加载到模型参数的策略，使上层加载流程与底层存储解耦。这里比较常见的就是DefaultModelLoader。
- model_loader.load_model(...)：调用具体加载器执行模型加载，如上所述，其内部依次经历三个阶段：initialize_model、load_weights 和 process_weights_after_loading，最终返回一个经过 model.eval() 处理后的 nn.Module 对象。对应于上图的1~3。
def load_model(self, eep_scale_up: bool = False) -> None:
    """
        Args:
        eep_scale_up: the model loading is for elastic EP scale up.
    """
    logger.info("Starting to load model %s...", self.model_config.model)

    with DeviceMemoryProfiler() as m:
    time_before_load = time.perf_counter()
    model_loader = get_model_loader(self.load_config)
    logger.info("Loading model from scratch...")
    self.model = model_loader.load_model(vllm_config=self.vllm_config, model_config=self.model_config)
随后来看 load_model  的实现。该函数首先根据 device_config  和 load_config 确定目标加载设备，然后在 model_config.dtype 指定的数据类型环境下，依次完成模型结构实例化、权重加载和加载后处理，最后返回 eval 模式下的模型对象。
def load_model(self, vllm_config: VllmConfig,
               model_config: ModelConfig) -> nn.Module:
    """Load a model with the given configurations."""
    device_config = vllm_config.device_config
    load_config = vllm_config.load_config
    load_device = device_config.device if load_config.device is None else \
    load_config.device
    target_device = torch.device(load_device)
    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            # 1. 实例化模型架构
            model = initialize_model(vllm_config=vllm_config,
                                     model_config=model_config)

            logger.debug("Loading weights on %s ...", load_device)
            # Quantization does not happen in `load_weights` but after it
            # 2. 加载权重
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()
我们设置 tp = 4，表示在 4 张 GPU 上进行张量并行（Tensor Parallelism）。在此配置下，模型中的可切分权重会被均匀划分为 4 份，每张 GPU 负责其中一份分片，而非两份。除了 "可切分权重"（Linear、Embedding 等），LayerNorm、bias 等小参数会被完整复制到每个 rank 上，而非分片。
在 vLLM 执行架构中，Worker  是实际的工作单元，负责接收来自EngineProc的 RPC 请求，并调度 GPUModelRunner 完成模型加载与推理计算。
通常，每个 TP rank 对应一个独立的 Worker，因此在 4 路张量并行下，系统会启动 4 个 Worker，分别运行在 4 张 GPU 上。每个 Worker 内部持有一个 GPUModelRunner 实例，并仅加载对应本地 rank 的模型权重分片。从执行视角看，完整模型被分布式部署于多个 Worker 之间，各节点仅管理自身分片的权重与计算任务，通过协同完成整体推理流程。
简而言之：在 tp = 4 的设置下，模型权重被切分为 4 个分片，4 个 Worker 分别加载各自的分片，协同执行推理任务。
暂时无法在飞书文档外展示此内容
initialize_model
实例化模型架构，但是不包括权重加载的部分
暂时无法在飞书文档外展示此内容
这里首先要做的，是根据模型名称在 ModelRegistry 中定位对应的模型结构。对于 vLLM 来说，若开发者希望支持一种新模型，首先需要实现该模型对应的结构类；随后，再将“模型名称字符串”和“模型结构类”之间的映射关系注册到 ModelRegistry 中。这样，后续就可以通过类似上文中的'Qwen3ForCausalLM' 这样的字符串，动态找到并实例化对应的模型类。
1. 先读取 model_config.hf_config.architectures：vLLM 会从 Hugging Face 配置中的 architectures  字段提取模型结构名称，例如 Qwen3ForCausalLM。
2. 调用model_config.registry.resolve_model_cls(...)查找模型类：将架构名称列表传入 ModelRegistry，由其在内部注册表中查找对应的模型实现。对于 vLLM 原生支持的模型，定位到注册表即可命中；
3. 在 ModelRegistry.models 中定位注册项，而注册表内部维护一张映射表，键为模型架构名，值为注册项。对于 Qwen3ForCausalLM，系统将定位到对应的注册信息，其中至少包含：  
  - module_name：模型类所在模块路径，例如 vllm.model_executor.models.qwen3  
  - class_name：实际类名，例如 Qwen3ForCausalLM
4. 找到注册项后，vLLM 调用 load_model_cls()，内部通过 importlib.import_module(self.module_name) 动态导入对应模块，再使用 getattr(mod, self.class_name) 提取真正的 Python 类对象。以 Qwen3 为例，可以等价理解为：
mod = importlib.import_module("vllm.model_executor.models.qwen3")
model_class = getattr(mod, "Qwen3ForCausalLM")
5.  model = model_class(vllm_config=vllm_config, prefix=prefix)这一步会调用模型类的构造函数，完成模型实例化，并生成对应的模型对象。
model = model_class(vllm_config=vllm_config, prefix=prefix)
这里的查找与实例化流程可以概括为：vLLM 先从 model_config.hf_config.architectures 中读取模型架构名称，然后调用 ModelRegistry.resolve_model_cls() 在注册表中查找对应的注册项。注册项中记录了模型类所在的模块路径和类名，因此 vLLM 可以先动态导入相应模块，再取出目标模型类，最后由 initialize_model() 调用 model_class(vllm_config=vllm_config, prefix=prefix) 完成模型实例化，生成最终的模型对象。
暂时无法在飞书文档外展示此内容
def initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    model_class: Optional[type[nn.Module]] = None,
    model_config: Optional[ModelConfig] = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""
 
    model_class, _ = get_model_architecture(model_config)
    
def get_model_architecture(
        model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])
    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures,
        model_config=model_config,
    )
resolve_model_cls 返回一个元组：model_cls 是已解析的模型类对象（如 vllm.model_executor.models.qwen3.Qwen3ForCausalLM），arch 是匹配到的模型架构名称字符串（如 "Qwen3ForCausalLM"）。
(model_cls, arch)
在resolve_model_cls方法中：
1. 当获得 module_name 和 class_name 之后，vLLM 会先动态导入对应模块，再从模块中取出目标模型类。这里的 importlib.import_module 是 Python 标准库提供的动态导入接口，可以根据字符串形式的模块路径加载对应模块。
def load_model_cls(self) -> type[nn.Module]:
    mod = importlib.import_module(self.module_name)
    return getattr(mod, self.class_name)
2. 获取到模型类后，返回到 get_model_architecture 函数中。此时已掌握两个关键信息：一是模型的名称（architecture name arch），二是模型的类对象 model_cls（由 load_model_cls 返回）。
3. 这里得到的 model_cls 仍然只是一个类对象，还不是实际运行时的模型实例。真正的模型实例化会在 initialize_model() 的后续部分中完成：先根据量化配置调用 configure_quant_config，再调用模型类构造函数，传入 vllm_config、prefix 等必要参数，生成具体的模型对象。
暂时无法在飞书文档外展示此内容
注册新模型的过程
在上一节中我们在ModelRegistry中找到了qwen3的定义，那么这一节课中我们就来看看模型的定义是怎么被注册到ModelRegistry当中的，这是它在全局当中的实例化：
ModelRegistry = _ModelRegistry({
    model_arch:
    _LazyRegisteredModel(
        # module_name，表示模型类所在的模块路径（即 Python 文件路径）；
        # model_class_name，表示实际要实例化的模型类名。
        module_name=f"vllm.model_executor.models.{mod_relname}",
        class_name=cls_name,
    )
    # 可以看出模型项是放在_VLLM_MODELS中的，在初始化的时候插入其中
    for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
})
这里的三元组对大家来说应该非常熟悉：model_arch, (mod_relname, cls_name)。以上文中的例子为例，即为 ('Qwen3ForCausalLM', ('vllm.model_executor.models.qwen3', 'Qwen3ForCausalLM'))。在 ModelRegistry 实例化过程中，这些类信息会被注册到其内部的 models 变量中。该变量是一个字典（或映射结构），每一项对应一个这样的三元组，用于记录模型架构名称与其对应模块路径和类名之间的映射关系。
_VLLM_MODELS 的结构为 dict[model_arch, tuple[mod_relname, cls_name]]，其字段含义如下：
部分
示例
含义
model_arch
"Qwen3ForCausalLM"
Hugging Face config.json  中的 architectures 字段值，标识模型架构类型
mod_relname
"qwen3"
模块的相对路径名称，拼接后构成完整的模块路径：vllm.model_executor.models.qwen3
cls_name
"Qwen3ForCausalLM"
该模块中实际定义的模型类名，用于动态导入和实例化
该注册表是 vLLM 实现模型动态解析的核心基础，通过模型架构名快速定位到对应的模块路径与类名，从而实现从字符串名称到具体模型类的映射。
接下来我们查看 _VLLM_MODELS 中_TEXT_GENERATION_MODELS的定义，其中每一项正是这样一个二元组 (模块相对路径, 类名)。通过查找可以定位到 Qwen3 对应的注册项，这表明 Qwen3 的模型类在此处被正式注册到模型注册表中，供后续在模型加载时通过名称查找到对应的实现类并进行实例化访问。
_VLLM_MODELS = {
    **_TEXT_GENERATION_MODELS,
    **_EMBEDDING_MODELS,
    **_CROSS_ENCODER_MODELS,
    **_MULTIMODAL_MODELS,
    **_SPECULATIVE_DECODING_MODELS,
    **_TRANSFORMERS_SUPPORTED_MODELS,
    **_TRANSFORMERS_BACKEND_MODELS,
}

_TEXT_GENERATION_MODELS = {
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
}
以 Qwen3ForCausalLM 为例，它在 _TEXT_GENERATION_MODELS 中的注册项表示：
当 architectures 字段为 Qwen3ForCausalLM 时，vLLM 会到 vllm.model_executor.models.qwen3 模块中查找 Qwen3ForCausalLM 这个类。也就是说，Qwen3 的模型类在这里被注册进模型注册表，供后续模型加载时动态查找和实例化。如果需要支持一个新的模型，通常也需要在相应的注册表中加入新的映射项。但前提是已经实现了对应的模型类，并且该实现满足 vLLM 的模型接口要求，尤其是模型初始化、前向计算、权重加载和并行切分等逻辑。
让 vLLM 支持一个新的模型
我们的模型是一个示例模型，整体结构为单层 MLP，相关文件位于 code/course6 目录下。配置文件中的 "architectures" 字段对应 ModelRegistry 注册映射中的 key，用于唯一标识当前模型架构。例如这里的 "MLPModel"，后续就会被 vLLM 用来查找对应的模型实现类。
除 "architectures" 外，配置中还包含模型运行所需的其他参数，例如 input_dim、hidden_dim 和 output_dim，分别用于定义 MLP 层的输入维度、隐藏层维度和输出维度。由于该示例模型只包含一个简单的 MLP 结构，因此无需处理多层网络或复杂模块之间的连接逻辑。
{
  "architectures": [
    "MLPModel"
  ],
  "dtype": "float32",
  "hidden_dim": 256,
  "input_dim": 128,
  "model_type": "llama",
  "output_dim": 10,
  "pad_token_id": -1,
  "transformers_version": "4.56.1",
  "vocab_size": 1024
}
模型的定义如下， 它由两个Linear层和一个激活算子ReLU组成，分别是self.fc1和self.fc2和self.relu。
class MLPConfig(PretrainedConfig):
    model_type = "llama"  
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.vocab_size = 1024
        self.pad_token_id = -1

from transformers import PreTrainedModel
import torch.nn as nn

class MLPModel(PreTrainedModel):
    config_class = MLPConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.input_dim,
            padding_idx=config.pad_token_id,
        )
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x):
        x = self.fc1(self.embed_tokens(x))
        x = self.relu(x)
        return self.fc2(x)
随后需要将模型导出为 Hugging Face 格式。这样做的原因是 vLLM 原生支持 Hugging Face 模型目录结构，后续我们也会基于该格式分析 vLLM 如何查找配置文件、解析模型结构并加载权重。导出流程主要分为三步：
1. 实例化模型配置 config，用于定义模型的基本参数和结构信息；
2. 创建模型实例，并调用 save_pretrained() 将模型配置和权重保存为 Hugging Face 格式；
3. 导出 tokenizer 相关文件。虽然当前示例模型并不会真正使用该 tokenizer 进行分词，但 vLLM 初始化流程通常仍会读取 tokenizer 配置，因此这里使用一个最小 tokenizer 作为占位，以保持目录结构完整。
config = MLPConfig(input_dim=128, hidden_dim=256, output_dim=10)
model = MLPModel(config)

# 保存为 Hugging Face 格式
model.save_pretrained("/home/test_fss/code/vllm_learn/code/course6/mlp_model")

from transformers import AutoTokenizer

# 用个最小的 tokenizer 占位
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("/home/test_fss/code/vllm_learn/code/course6/mlp_model")
Hugging Face 格式的模型导出完成后，还需要在 vLLM 端添加对应的模型实现，具体可参考 vllm/model_executor/models/mlp.py。
该文件中的模型结构需要与导出阶段保持一致，包括 Embedding 层、两个 Linear 层以及一个 ReLU 激活函数。这样，vLLM 在实例化模型结构后，才能正确加载 checkpoint 中保存的权重。这里的词表大小实际对应模型中的 self.embed_tokens 模块。它的作用是将输入的 token_ids 映射为连续向量，作为后续 MLP 的输入表示。
当 models/mlp.py 中的模型类实现完成后，还需要将其注册到全局 ModelRegistry 中。同时，注册项中的 key 必须与 config.json 里的 architectures 字段一致，否则 vLLM 无法根据架构名称找到对应的模型类。
例如，如果配置文件中写的是：
"architectures": ["MLPForCausalLM"]
那么注册项应写为：
_TEXT_GENERATION_MODELS = {
    "MLPForCausalLM": ("mlp", "MLPForCausalLM"),
}
这样，在模型加载时，vLLM 就可以根据 MLPForCausalLM 定位到 vllm.model_executor.models.mlp 模块，并实例化其中的 MLPForCausalLM 类。

model_executor/models/mlp.py 中需要新增 vLLM 侧的模型定义。其中，MLPModel 负责实现实际的网络结构，MLPForCausalLM 则作为对外注册和实例化的模型入口。后续 vLLM 在模型加载时，会根据 config.json 中的 architectures 字段找到 MLPForCausalLM，并通过该类构造出完整模型。
class MLPForCausalLM(nn.Module):
    pass
    
class MLPModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        """
            self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(config.hidden_dim, config.output_dim)
        """
        self.embed_tokens = VocabParallelEmbedding(
            cfg.vocab_size,
            cfg.input_dim,
            quant_config=vllm_config.quant_config,
        )

        self.fc1 = RowParallelLinear(
            cfg.input_dim, cfg.hidden_dim, bias=True, quant_config=quant_config
        )
        self.act = nn.ReLU()
        self.fc2 = RowParallelLinear(
            cfg.hidden_dim, cfg.output_dim, bias=True, quant_config=quant_config
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        x = self.embed_tokens(input_ids)
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x
模型权重的加载流程
从图一可以看出，在张量并行场景下，每个 GPUModelRunner 通常对应一个 TP rank，并负责加载当前 rank 所需的模型权重分片。因此，权重加载流程可以从 GPUModelRunner.load_model() 开始分析。该方法本身主要负责组织加载流程，真正的模型实例化与权重加载则发生在 model_loader.load_model() 中。
class GPUModelRunner:
    def load_model(self, eep_scale_up: bool = False) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
    
        with DeviceMemoryProfiler() as m:
             model_loader = get_model_loader(self.load_config)
             logger.info("Loading model from scratch...")
             self.model = model_loader.load_model(vllm_config=self.vllm_config, model_config=self.model_config)
随后进入之前分析过的 BaseModelLoader 类中的 load_model 方法。我们先回顾一下：在 initialize_model 中，会根据模型名称从注册表 ModelRegistry 中查找对应的模块及模型类定义，并结合配置文件中的参数实例化该模型类。
class BaseModelLoader(ABC):
    def load_model(self, vllm_config: VllmConfig,
                         model_config: ModelConfig) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = device_config.device if load_config.device is None else \
                     load_config.device
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)
            # 开始模型权重的加载流程
            self.load_weights(model, model_config)
暂时无法在飞书文档外展示此内容
随后流程进入前面分析过的 BaseModelLoader.load_model()。这里先简单回顾一下：在 initialize_model() 阶段，vLLM 会读取配置文件中的 architectures 字段，得到模型架构名称；然后通过 ModelRegistry 查找对应的模块路径和模型类定义，并结合 vllm_config、model_config 等配置完成模型实例化，注意，这一步只是搭建模型结构，当模型结构创建完成后，流程才会继续进入 load_weights()，开始真正加载 checkpoint 权重。
其中的DefaultModelLoader是BaseModelLoader的一个子类。
class DefaultModelLoader(BaseModelLoader):
    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        weights_to_load = {name for name, _ in model.named_parameters()}
        loaded_weights = model.load_weights(
            self.get_all_weights(model_config, model))
        self.counter_after_loading_weights = time.perf_counter()
    
        if model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError("Following weights were not initialized from "
                                 f"checkpoint: {weights_not_loaded}")
例如，当权重文件为 safetensors 格式时，会使用 safetensors_weights_iterator() 或 fastsafetensors_weights_iterator() 逐个产出 checkpoint 中的权重项。因此，这一阶段的核心作用不是完成参数加载，而是把不同格式、不同来源的 checkpoint 权重统一转换为 (name, tensor) 形式，并交给后续的模型级 load_weights() 处理。
也就是说，DefaultModelLoader.get_all_weights() 负责定位并读取 checkpoint 权重文件，生成 (name, tensor) 形式的权重迭代器；真正把这些权重写入模型参数的逻辑发生在后续的 model.load_weights(iterator) 中。
class DefaultModelLoader(BaseModelLoader):
    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path, source.revision, source.fall_back_to_pt,
            source.allow_patterns_overrides)
        ...
        ...
        if use_safetensors:
            if self.load_config.load_format == "fastsafetensors":
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            else:
                weights_iterator = safetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
                
    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load",
                                    True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides",
                                             None),
        )
        yield from self._get_weights_iterator(primary_weights)
1. 首先，DefaultModelLoader.load_weights() 会通过 model.named_parameters() 获取当前模型中所有需要加载的参数名：
weights_to_load = {name for name, _ in model.named_parameters()}
这一步主要用于记录模型期望加载哪些参数，后续可以据此检查权重是否完整加载。
2. 接着，根据 model_config 中的模型路径构造权重来源，并调用 get_all_weights() 获取 checkpoint 权重迭代器。其内部会进入yield from self._get_weights_iterator(primary_weights)
yield from self._get_weights_iterator(primary_weights)
  _get_weights_iterator() 会先定位模型目录和权重文件，得到 hf_folder 和 hf_weights_files。例如，在加载 Qwen3-1.7B 时，hf_weights_files 可能包含两个 safetensors 分片文件：
['/home/test_fss/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86cc...908b1ad5e/model-00001-of-00002.safetensors', 
'/home/test_fss/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86cc...908b1ad5e/model-00002-of-00002.safetensors']
  随后，vLLM 会根据权重文件类型选择不同的迭代器。对于 safetensors 文件，会使用 safetensors_weights_iterator() 或 fastsafetensors_weights_iterator()。该迭代器每次产出一个 (name, tensor) 对，其中 name 是 checkpoint 中的权重名称，tensor 是对应的 PyTorch 权重张量。
  需要注意的是，此时产出的 tensor 通常仍位于 CPU 侧。后续 DefaultModelLoader.load_weights() 会把这些权重传给 model.load_weights()；在这里，vLLM 会完成参数名匹配，并根据参数绑定的 weight_loader 决定是否按 TP rank 对权重做切分，然后再写入目标参数。
3. 通过前面的步骤我们已经获得了模型 checkpoint 的权重迭代器。使用迭代器的好处是，vLLM 不需要一次性将所有权重文件完整加载到内存中，而是可以逐个产出 (name, tensor) 并依次加载，从而降低加载阶段的内存峰值。随后，DefaultModelLoader.load_weights() 会将该迭代器传入模型自身的 load_weights() 方法。以 Qwen3ForCausalLM 为例：
class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
  - AutoWeightsLoader 可以理解为一个按权重名称进行分发的调度器。它不会直接处理 q_proj -> qkv_proj 这类 fused 参数映射，而是先根据 checkpoint 中的权重名前缀，判断这块权重应该交给哪个子模块或参数。例如，model.layers.0.self_attn.q_proj.weight 会先根据 model 前缀被路由到 Qwen3ForCausalLM 的 self.model，也就是 Qwen3Model。
  - 如果目标子模块自身实现了 load_weights()，AutoWeightsLoader 会将对应权重继续交给该子模块处理；因此在 Qwen3 中，最终会进入 Qwen3Model 继承自 Qwen2Model 的 load_weights()，并在那里完成 q_proj -> qkv_proj、gate_proj -> gate_up_proj 等 fused 参数映射与分片加载。
  - 否则，对于普通参数，AutoWeightsLoader 则会继续按名称找到对应的 parameter，并调用其 weight_loader 或默认加载函数default_weight_loader完成拷贝。
[图片]
  因此，Qwen3 加载权重的完整调用链可以概括为：
  1. DefaultModelLoader.load_weights() 获取 (name, tensor) 权重迭代器；
def load_weights(self, model, model_config) -> None:
    weights_to_load = {name for name, _ in model.named_parameters()}
    loaded_weights = model.load_weights(
        self.get_all_weights(model_config, model)
    )
  2. 调用顶层模型 Qwen3ForCausalLM.load_weights(weights)；
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DecoderLayer,
        )


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = Qwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

    def load_weights(self, weights):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
  3. Qwen3ForCausalLM 通过 AutoWeightsLoader 从顶层模型开始自动分发权重。
    在 AutoWeightsLoader.load_weights() 中，self._load_module("", self.module, weights) 会从 Qwen3ForCausalLM 这个顶层模块开始解析权重名。它会先读取当前模块的直接子模块，例如对于 Qwen3ForCausalLM，child_modules可以理解为：
{
    "model": Qwen3Model(...),
    "lm_head": ParallelLMHead(...),
    "logits_processor": LogitsProcessor(...),
}
    随后，AutoWeightsLoader 会按照权重名的第一个 . 进行拆分和分组。以 model.layers.0.self_attn.q_proj.weight 为例，它会被拆成第一级前缀 model，以及剩余部分 layers.0.self_attn.q_proj.weight。因此，对应的 child_weights 可以理解为：
child_weights = [
    ("layers.0.self_attn.q_proj.weight", tensor_q),
    ("layers.0.self_attn.k_proj.weight", tensor_k),
]
    由于第一级前缀 model 能在 child_modules 中找到，AutoWeightsLoader 就会把这组去掉 model. 前缀后的 child_weights 递归分发给 child_modules["model"]，也就是 self.model = Qwen3Model(...)。 进入 Qwen3Model 后，由于它继承自 Qwen2Model，最终会继续调用 Qwen2Model.load_weights()，完成主干权重加载以及后续的 fused 参数映射。
class AutoWeightsLoader:
    def load_weights(self, weights, *, mapper=None) -> set[str]:
        if mapper is not None:
            weights = mapper.apply(weights)

        weights = (
            (name, weight)
            for name, weight in weights
            if not self._can_skip(name)
        )

        autoloaded_weights = set(
            self._load_module("", self.module, weights)
        )
        return autoloaded_weights
  4. 由于Qwen3Model 继承自 Qwen2Model，因此会复用 Qwen2Model.load_weights()，而在 Qwen2Model.load_weights() 中，完成 qkv_proj、gate_up_proj 等 fused 参数映射与 TP 分片加载。 weight_loader 负责真正把 loaded_weight 写入目标参数 param。对于普通参数，它通常直接调用默认逻辑，将 checkpoint 中的权重拷贝到模型参数中。对于 fused 参数或 TP 并行参数，它还需要根据 shard_id 判断当前权重属于哪一段，例如 q/k/v，再结合当前 TP rank 计算读取范围和写入位置，最后把对应分片写入融合后的目标参数中。
def load_weights(self, weights) -> set[str]:
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    params_dict = dict(self.named_parameters(remove_duplicate=False))
    loaded_params = set()

    for name, loaded_weight in weights:
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue

            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            weight_loader = getattr(
                param, "weight_loader", default_weight_loader
            )

            if weight_loader == default_weight_loader:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)
                break

    return loaded_params
    这里的关键点是：Qwen3ForCausalLM 是 vLLM 实例化的顶层模型类，负责封装 self.model、lm_head 和 logits 处理逻辑；Qwen3Model 是实际的 Transformer 主干，而AutoWeightsLoader 主要负责按前缀把权重路由到对应子模块，真正的 fused 参数映射和 weight_loader 调用发生在 Qwen2Model.load_weights() 中。
  5. Qwen2Model 中的算子可以分为两类：一类称为融合算子，另一类称为非融合算子。对于融合算子而言，多个原始算子的权重会被合并加载到单个算子中，并在加载过程中考虑模型并行（TP）的分片策略。
    非融合算子的权重通常可以按照参数名直接加载；而融合算子则需要将 checkpoint 中多个原始参数映射并合并到同一个 vLLM 算子中，同时在加载过程中结合张量的分片策略写入当前 rank 对应的参数区域。以 Qwen 模型中的 qkv_proj 为例，它会将输入激活一次性投影为 query、key 和 value 三个部分。因此，该算子的输出维度可以表示为：
[batch_size, seq_len, (num_heads + 2 * num_kv_heads) * head_dim]
    随后在 forward() 中通过 split() 将其拆分为 q、k、v 三部分。也就是说，vLLM 内部不再分别执行 q_proj、k_proj、v_proj 三个线性层，而是使用一个融合后的 QKVParallelLinear 一次性计算Q、K、V。相比三次独立 GEMM，这种方式通常更高效，因为它减少了 CUDA kernel 启动次数，也避免了对同一份输入激活的重复读取。
    需要注意的是，checkpoint 中的权重仍然可能按照 HuggingFace 原始结构保存为 q_proj.weight、k_proj.weight、v_proj.weight 三个独立权重，而 vLLM 内部使用的是融合后的 qkv_proj.weight。
    因此，在加载权重时，Qwen2Model.load_weights() 会将 q_proj/k_proj/v_proj 映射到 qkv_proj，并分别传入 shard_id="q"、"k"、"v" 调用 weight_loader(param, loaded_weight, shard_id)。这里的 weight_loader 会根据 shard_id、TP rank 以及 KV head 的切分或复制规则，把当前权重写入 qkv_proj.weight 中对应的 Q、K、V 区域。
weight_loader(param, loaded_weight, shard_id)
    这里的 shard_id = "q"，表示把当前的 q_proj.weight 加载到 qkv_proj.weight 的 Q 区域。
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        # (vLLM内部融合参数名, checkpoint中的原始参数名, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    params_dict = dict(self.named_parameters(remove_duplicate=False))
    loaded_params: set[str] = set()

    for name, loaded_weight in weights:
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue

            # 例如 self_attn.q_proj.weight -> self_attn.qkv_proj.weight
            name = name.replace(weight_name, param_name)

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)

            if weight_loader == default_weight_loader:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)

            break

    return loaded_params
    具体步骤可分为：
    - for name, loaded_weight in weights：遍历 checkpoint 权重，每次取出一个权重名和对应张量。
    - 对于 q_proj、k_proj、v_proj、gate_proj、up_proj 等需要融合的权重，先通过 stacked_params_mapping 将其映射到 vLLM 内部的融合参数名，例如 q_proj -> qkv_proj、q_proj -> qkv_proj、gate_proj -> gate_up_proj。
    - 根据映射后的参数名，从 params_dict 中找到目标参数 param。
    - 调用 weight_loader(param, loaded_weight, shard_id) 完成加载。这里的 shard_id 用来标识当前权重属于融合算子的哪一段，例如 "q"、"k"、"v"。也就是从checkpoint 中读出的 loaded_weight，加载到当前模型参数 param 中。
    - 在 weight_loader 内部，会根据 shard_id 和当前 TP rank 计算读取范围与写入位置，然后将对应权重分片拷贝到目标参数中。
    
    在 fused 参数加载时，vLLM 会先根据当前加载的是 q、k 还是 v，计算该分支在融合参数中的写入范围。对于 qkv_proj 来说，当前 rank 本地的目标参数仍然按照 [Q | K | V] 的顺序组织，因此不同分支会写入不同的区域：
    - q -> 写入 Q 段
    - k -> 写入 K 段
    - v -> 写入 V 段
    其中，shard_offset 表示该分支在本地融合参数中的写入起点，shard_size 表示该分支在当前 TP rank 上需要写入的长度。确定目标写入范围后，vLLM 再根据当前 tp_rank 计算 start_idx，从 checkpoint 原始权重中切出当前 rank 负责的那一段：
start_idx = tp_rank * shard_size
    最后，将切出的权重分片写入融合参数的对应区域。整个过程可以理解为：先确定写到融合参数的哪一段，再确定从原始权重中读取哪一段，最后完成分片拷贝。
class QKVParallelLinear(ColumnParallelLinear):
    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        # 根据 shard_id 判断当前加载的是 q/k/v 中的哪一块，
        # 并计算它在融合参数 qkv_proj.weight 中的目标区间。
        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == "k":
            shard_offset = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        elif loaded_shard_id == "v":
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
            shard_size = self.num_kv_heads * self.v_head_size

        # 在融合后的目标参数中，选中当前 q/k/v 对应的写入区域。
        param_data = param.data
        param_data = param_data.narrow(output_dim, shard_offset, shard_size)

        # q head 按 TP rank 切分；k/v 在 GQA/MQA 下可能会被多个 TP rank 复用。
        if loaded_shard_id == "q":
            shard_rank = self.tp_rank
        else:
            shard_rank = self.tp_rank // self.num_kv_head_replicas

        # 从 checkpoint 权重中取出当前 rank 需要加载的那一段。
        start_idx = shard_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # 写入 qkv_proj.weight 的对应区域。
        param_data.copy_(loaded_weight)
    shard_size 表示当前分支在当前 TP rank 上需要写入的宽度。例如对于 qkv_proj，它表示 Q、K 或 V 中某一个分支在当前 rank 上对应的参数宽度。也就是说，shard_offset 决定写到融合参数的哪一段，shard_size 决定这一段写多长。
暂时无法在飞书文档外展示此内容
    以加载 k_proj.weight 为例，假设 checkpoint 中：
q_proj.weight = [q0, q1, q2, q3, q4, q5, q6, q7]
k_proj.weight = [k0, k1, k2, k3]
v_proj.weight = [v0, v1, v2, v3]
    当前 tp_size = 2，并且当前 worker 是 rank = 1。因此每个 rank 只加载各自负责的一半权重。
    rank 0 加载：
q_proj -> [q0, q1, q2, q3]
k_proj -> [k0, k1]
v_proj -> [v0, v1]
    rank 1 加载：
q_proj -> [q4, q5, q6, q7]
k_proj -> [k2, k3]
v_proj -> [v2, v3]
    根据公式param_data[shard_offset:shard_offset + shared_size] = loaded_weight[start_idx:start_idx + shared_size]，对于当前 rank = 1 来说，加载 k_proj.weight 时，需要从 checkpoint 的 k_proj.weight 中读取 [k2, k3]，因此：
start_idx = 2
shard_size = 2
    而 vLLM 内部的 qkv_proj.weight 是融合后的参数，在当前 rank 本地可以理解为：
qkv_proj.weight = [Q 区域         | K 区域 | V 区域]
                  [q4 q5 q6 q7    | _ _ _ _| _ __ _]
    由于当前 rank 本地 Q 区域长度为 4，所以 K 区域从位置 4 开始：shard_offset = 4，最终加载可以理解为：param_data[4:6] = loaded_weight[2:4]
    在这个例子中：
    - start_idx = 2
    - shard_offset = 4
    - shard_size = 2
    所以：
loaded_weight[start_idx : start_idx + shard_size]
= loaded_weight[2 : 2 + 2]
= loaded_weight[2 : 4]
    表示从 checkpoint 原始权重中读取当前 TP rank 负责的分片。
param_data[shard_offset : shard_offset + shard_size]
= param_data[4 : 4 + 2]
= param_data[4 : 6]
    表示将这段分片写入融合后参数的目标区域。
    - start_idx 决定从 loaded_weight 的哪里开始读；
    - shard_offset 决定写到 param_data 的哪里；
    - shard_size 决定读取和写入的长度。
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(self.named_parameters(remove_duplicate=False))
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue
        for (param_name, weight_name, shard_id) in stacked_params_mapping:
            # 如果是融合后的算子
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name.endswith("scale"):
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            # 确定要写入的位置
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            if weight_loader == default_weight_loader:
                weight_loader(param, loaded_weight)
            else:
                # 加载融合算子
                weight_loader(param, loaded_weight, shard_id)
            break
自定义模型权重加载函数
我们之前已经让 vLLM 支持了一个新的示例模型。为了让该模型能够正确加载 Hugging Face safetensors 格式的权重，需要为 vLLM 侧的 MLPModel 或其外层封装类实现load_weights() 方法，用来自定义 checkpoint 权重到模型参数之间的映射关系
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[st
    found_params = set()
    for name, loaded_weight in weights:
        if name == "fc1.weight":
            param = self.model.fc1.weight
            set_weight_attrs(loaded_weight, {"param_name": "model.fc1.weight"
            default_weight_loader(param, loaded_weight)
            name = "model." + name
            found_params.add(name)
        
        ...
        ...
    return found_params
编写 load_weights() 时，需要注意两点。
1. 首先，对于已经成功处理的权重，需要将对应的模型参数名记录到 found_params 中，并在函数最后返回。调用方会根据这个返回值进行校验，以确认模型中应加载的参数是否都已经正确加载。
2. 其次，由于这个 MLP 示例模型没有使用 fused 参数，因此不需要处理类似 q_proj -> qkv_proj 或 gate_proj -> gate_up_proj 这样的映射逻辑。对于普通权重，只需要找到目标参数 param，然后调用 default_weight_loader(param, loaded_weight) 即可。其核心逻辑可以理解为：
param.data.copy_(loaded_weight)
也就是将 checkpoint 中读取出的 loaded_weight 拷贝到 vLLM 模型内部对应的参数中。如果目标参数位于 GPU，而 loaded_weight 来自 CPU，这一步就会完成 CPU 到 GPU 的权重拷贝。另外，这里返回模型中所有已加载算子的名称，是因为调用方需要进行校验，以确保该模型中所有算子的权重都已被正确加载。
总结
vLLM模型加载流程始于ModelRunner.load_model，通过model_loader.load_model实例化模型架构并加载权重。
1. 首先，initialize_model根据ModelConfig从ModelRegistry中查找并实例化对应模型类；
2. 注册表通过_VLLM_MODELS管理支持的模型，新增模型需在此注册。
3. 随后，load_weights遍历权重文件（如safetensors），利用生成器逐层加载，AutoWeightsLoader调度各子模块的load_weights方法。
4. 对于融合算子（如Qwen的qkv_proj），通过stacked_params_mapping将多个权重合并，并按TP分片写入指定位置。
5. 自定义模型如果需要实现load_weights，可以使用默认的权重加载方式default_weight_loader，该方法会返回已加载参数名以供校验，确保所有参数正确加载。
