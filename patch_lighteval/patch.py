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

