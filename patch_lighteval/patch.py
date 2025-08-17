from importlib import import_module
import importlib
import inspect
import textwrap
import sys


def patch_reasoning():
    try:
        prompt_manager_module = import_module("lighteval.tasks.prompt_manager")
        pipeline_module = import_module("lighteval.pipeline")
        lighteval_task_module = import_module("lighteval.tasks.lighteval_task")

        create_requests_from_tasks = getattr(
            lighteval_task_module, "create_requests_from_tasks", None
        )
        Pipeline = getattr(pipeline_module, "Pipeline", None)
        PromptManager = getattr(prompt_manager_module, "PromptManager", None)

        # define the functions to patch
        def patch_single_turn_context():
            _single_turn_context = getattr(PromptManager, "_single_turn_context", None)

            function = inspect.getsource(_single_turn_context)
            function = textwrap.dedent(function)

            old_code = (
                "return self.model.tokenizer.apply_chat_template(\n"
                "            output, tokenize=False, add_generation_prompt=True\n"
                "        ), num_effective_fewshots"
            )

            new_code = (
                "return self.model.tokenizer.apply_chat_template(\n"
                "            output, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking\n"
                "        ), num_effective_fewshots"
            )

            function = function.replace(old_code, new_code)
            return function

        def patch_PromptManager_init():
            init = getattr(PromptManager, "__init__", None)
            prompt_init = inspect.getsource(init)
            prompt_init = textwrap.dedent(prompt_init)

            old_signature = (
                'def __init__(self, task: "LightevalTask", lm: LightevalModel):'
            )
            new_signature = 'def __init__(self, task: "LightevalTask", lm: LightevalModel, **kwargs):'

            prompt_init = prompt_init.replace(old_signature, new_signature)

            insertion = (
                '    self.enable_thinking = kwargs.get("enable_thinking", False)'
            )
            prompt_init = prompt_init.replace(
                "    self.few_shot_sampler = FewShotSampler(task)",
                "    self.few_shot_sampler = FewShotSampler(task)\n" + insertion,
            )

            return prompt_init

        def patch_pipeline(patched_create_requests_from_tasks):
            class_code = inspect.getsource(Pipeline)
            class_code = textwrap.dedent(class_code)

            old_signature = (
                "def __init__(\n"
                "        self,\n"
                "        tasks: str,\n"
                "        pipeline_parameters: PipelineParameters,\n"
                "        evaluation_tracker: EvaluationTracker,\n"
                "        model_config=None,\n"
                "        model=None,\n"
                "        metric_options=None,\n"
                "    ):"
            )

            new_signature = (
                "def __init__(\n"
                "        self,\n"
                "        tasks: str,\n"
                "        pipeline_parameters: PipelineParameters,\n"
                "        evaluation_tracker: EvaluationTracker,\n"
                "        model_config=None,\n"
                "        model=None,\n"
                "        metric_options=None,\n"
                "        **kwargs\n"
                "    ):"
            )

            class_code = class_code.replace(old_signature, new_signature)
            injection = (
                '        self.enable_thinking = kwargs.get("enable_thinking", False)'
            )
            if injection not in class_code:
                insertion_point = "self.model = self._init_model(model_config, model)"
                class_code = class_code.replace(
                    insertion_point, f"{insertion_point}\n{injection}"
                )

            old_call = (
                "requests, docs = create_requests_from_tasks(\n"
                "                task_dict=task_dict,\n"
                "                fewshot_dict=fewshots_dict,\n"
                "                num_fewshot_seeds=self.pipeline_parameters.num_fewshot_seeds,\n"
                "                lm=self.model,\n"
                "                max_samples=self.pipeline_parameters.max_samples,\n"
                "                evaluation_tracker=self.evaluation_tracker,\n"
                "                use_chat_template=self.pipeline_parameters.use_chat_template,\n"
                "                system_prompt=self.pipeline_parameters.system_prompt,\n"
                "                cot_prompt=self.pipeline_parameters.cot_prompt,\n"
                "            )"
            )

            new_call = (
                "requests, docs = create_requests_from_tasks(\n"
                "                task_dict=task_dict,\n"
                "                fewshot_dict=fewshots_dict,\n"
                "                num_fewshot_seeds=self.pipeline_parameters.num_fewshot_seeds,\n"
                "                lm=self.model,\n"
                "                max_samples=self.pipeline_parameters.max_samples,\n"
                "                evaluation_tracker=self.evaluation_tracker,\n"
                "                use_chat_template=self.pipeline_parameters.use_chat_template,\n"
                "                system_prompt=self.pipeline_parameters.system_prompt,\n"
                "                cot_prompt=self.pipeline_parameters.cot_prompt,\n"
                "                enable_thinking=self.enable_thinking\n"
                "            )"
            )

            class_code = class_code.replace(old_call, new_call)
            local_ns = {}
            class_globals = Pipeline.__init__.__globals__.copy()
            # Add the patched create_requests_from_tasks to the globals
            class_globals["create_requests_from_tasks"] = (
                patched_create_requests_from_tasks
            )
            exec(class_code, class_globals, local_ns)
            patched_class = local_ns["Pipeline"]

            return patched_class

        def patch_create_requests_from_tasks():
            function = inspect.getsource(create_requests_from_tasks)
            function = textwrap.dedent(function)

            function = function.replace(
                "cot_prompt: str | None,\n)", "cot_prompt: str | None,\n    **kwargs\n)"
            )

            injection = '    enable_thinking = kwargs.get("enable_thinking", False)'
            if injection not in function:
                insertion_point = "requests: dict[RequestType, list[Request]] = collections.defaultdict(list)"
                function = function.replace(
                    insertion_point, f"{insertion_point}\n{injection}"
                )

            old_code = "        prompt_manager = PromptManager(lm=lm, task=task)"
            new_code = "        prompt_manager = PromptManager(lm=lm, task=task, enable_thinking=enable_thinking)"
            function = function.replace(old_code, new_code)

            return function

        # Patch the all necessary functions and classes
        # First patch the PromptManager functions
        function_single_turn_context = patch_single_turn_context()
        original_function = getattr(PromptManager, "_single_turn_context")
        local_ns = {}
        exec(function_single_turn_context, original_function.__globals__, local_ns)
        PromptManager._single_turn_context = local_ns["_single_turn_context"]
        sys.modules[
            "lighteval.tasks.prompt_manager"
        ].PromptManager._single_turn_context = PromptManager._single_turn_context

        function_init = patch_PromptManager_init()
        original_init = getattr(PromptManager, "__init__")
        local_ns_init = {}
        exec(function_init, original_init.__globals__, local_ns_init)
        PromptManager.__init__ = local_ns_init["__init__"]
        sys.modules["lighteval.tasks.prompt_manager"].PromptManager.__init__ = (
            PromptManager.__init__
        )

        # Then patch create_requests_from_tasks
        function_create_requests_from_tasks = patch_create_requests_from_tasks()
        local_ns = {}
        exec(
            function_create_requests_from_tasks,
            create_requests_from_tasks.__globals__,
            local_ns,
        )
        patched_create_requests_from_tasks = local_ns["create_requests_from_tasks"]

        # Apply the patched function to all relevant modules
        setattr(
            lighteval_task_module,
            "create_requests_from_tasks",
            patched_create_requests_from_tasks,
        )
        sys.modules["lighteval.tasks.lighteval_task"].create_requests_from_tasks = (
            patched_create_requests_from_tasks
        )
        pipeline_module.create_requests_from_tasks = patched_create_requests_from_tasks
        sys.modules["lighteval.pipeline"].create_requests_from_tasks = (
            patched_create_requests_from_tasks
        )

        # Finally patch the Pipeline class, passing the patched create_requests_from_tasks
        patched_pipeline = patch_pipeline(patched_create_requests_from_tasks)
        pipeline_module.Pipeline = patched_pipeline
        sys.modules["lighteval.pipeline"].Pipeline = patched_pipeline

        print("Lighteval enable_thinking successfully patched.")

    except Exception as e:
        raise RuntimeError(f"Failed to patch enable_thinking: {str(e)}")




from typing import Coroutine, Optional
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from vllm import LLM
def _create_auto_model(self, config: VLLMModelConfig) -> Optional[LLM]:
    """
    Creates an instance of the pretrained HF model.

    Args:
        pretrained (str): The name or path of the pretrained model.
        revision (str): The revision of the model.
        subfolder (Optional[str], optional): The subfolder within the model. Defaults to None.
        max_memory (Optional[dict], optional): The maximum memory to allocate for the model per GPU. Defaults to None.
        device_map (Optional[dict], optional): The device mapping for the model. Defaults to None.
        torch_dtype (Optional[Union[str, torch.dtype]], optional): The torch data type for the model. Defaults to None.
        quantization_config (Optional[Union[BitsAndBytesConfig, GPTQConfig]], optional): The quantization configuration for the model. Defaults to None.
        trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
        cache_dir (str, optional): The cache directory for the model. Defaults to "/scratch".

    Returns:
        transformers.PreTrainedModel: The created auto model instance.
    """
    self.model_args = {
        "model": config.model_name,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
        "dtype": config.dtype,
        "trust_remote_code": config.trust_remote_code,
        "tensor_parallel_size": config.tensor_parallel_size,
        "pipeline_parallel_size": config.pipeline_parallel_size,
        "max_model_len": self._max_length,
        "swap_space": 4,
        "seed": int(config.seed),
        "max_num_seqs": int(config.max_num_seqs),
        "max_num_batched_tokens": int(config.max_num_batched_tokens),
        "enable_prefix_caching": False        
        }

    if config.quantization is not None:
        self.model_args["quantization"] = config.quantization
    if config.load_format is not None:
        self.model_args["load_format"] = config.load_format

    if config.data_parallel_size > 1:
        self.model_args["distributed_executor_backend"] = "ray"
        self._batch_size = "auto"
        return None

    model = LLM(**self.model_args)

    # If the max_length can't get extracted from the config, it will be inferred from the model
    # Inferring from the tokenizer will cause vllm to bug for models with mismatches between model
    # config and tk config, like mistralai/Mistral-7B-v0.1
    if self._max_length is None:
        self._max_length = model.llm_engine.model_config.max_seq_len_to_capture

    return model

def patch_prefix_caching():
    try:
        from lighteval.models.vllm.vllm_model import VLLMModel
        
        function = inspect.getsource(_create_auto_model)
        function = textwrap.dedent(function)
        original_function = getattr(VLLMModel, "_create_auto_model")
        local_ns = {}
        exec(function, original_function.__globals__, local_ns)
        VLLMModel._create_auto_model = local_ns["_create_auto_model"]
        sys.modules[
            "lighteval.models.vllm.vllm_model"
        ].VLLMModel._create_auto_model = VLLMModel._create_auto_model

        print("Lighteval prefix caching successfully patched.")
    
    except Exception as e:
        raise RuntimeError(f"Failed to patch prefix caching: {str(e)}")
