from lighteval.metrics.metrics_sample import (
    PassAtK,
)
from lighteval.utils.language import Language
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    compare_gold_target, 
    extract_target_from_pred,
    get_extraction_regexes,
    multilingual_extractive_match_metric,
)
import numpy as np

from lighteval.metrics.metrics_sample import ExactMatches
import re
from typing import Callable
import os
from aenum import Enum
import numpy as np
from lighteval.metrics.normalizations import (helm_normalizer)
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import (
    Metric,
    MetricGrouping,
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric
)


# Metric for math-hard-fr task
math_pass_fr_at_1_1n = SampleLevelMetric(
    metric_name="math_pass_fr@1:1_samples",
    sample_level_fn=PassAtK(
        k=1,
        n=1,
        strip_strings=True,
        # Extracting mathematical expressions and latex expressions
        normalize_gold=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[
                    ExprExtractionConfig(),
                    LatexExtractionConfig(boxed_match_priority=0),
                ],
                language=Language.FRENCH,
            ),
        ),
        # Extracting mathematical expressions and latex expressions
        normalize_pred=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[
                    ExprExtractionConfig(),
                    LatexExtractionConfig(boxed_match_priority=0),
                ],
                language=Language.FRENCH,
            ),
        ),
        # Uses sympy for comparison
        sample_scoring_function=compare_gold_target,
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

# Metric for GPQA-Diamond-fr task
gpqa_instruct_pass_fr_at_1_1n = SampleLevelMetric(
    metric_name="gpqa_pass_fr@1:1_samples",
    sample_level_fn=PassAtK(
        k=1,
        n=1,
        sample_scoring_function=lambda pred, ref, doc: multilingual_extractive_match_metric(
            language=Language.FRENCH,
            gold_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters")
            ],
            pred_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters")
            ],
            precision=6,
        ).sample_level_fn(
            [ref], [pred], doc
        ),
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


class ExactMatchesThinking(ExactMatches):
    """
    A class to compute exact matches for reasoning tasks.
    This class extends the ExactMatches metric to handle reasoning-specific requirements.
    """

    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
        normalize_gold: Callable[[str], str] | None = None,
        normalize_pred: Callable[[str], str] | None = None,
        strip_strings: bool = False,
        type_exact_match: str = "full",
        answer_token: str = os.environ.get("answer_token", ""),  # for reasoning tasks we need to specify the answer token like <answer> or end of thinking token "</thinking>" if no asnwer token is outputted
    ):

        super().__init__(
            aggregation_function=aggregation_function,
            normalize_gold=normalize_gold,
            normalize_pred=normalize_pred,
            strip_strings=strip_strings,
            type_exact_match=type_exact_match,
        )
        self.answer_token = answer_token
    
    
    def compute_one_item(
        self,
        gold: str,
        pred: str,
    ) -> float:

        #extract the answer afte the answer token if it exists
        if self.answer_token:
            if self.answer_token in pred:
                pred = pred.split(self.answer_token, 1)[1]

        return super().compute_one_item(gold, pred)

class MetricsThinking:
    exact_match = SampleLevelMetric(
        metric_name="em",
        sample_level_fn=ExactMatchesThinking(strip_strings=True).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

    quasi_exact_match = SampleLevelMetric(
        metric_name="qem",
        sample_level_fn=ExactMatchesThinking(
            normalize_gold=helm_normalizer,
            normalize_pred=helm_normalizer,
            strip_strings=True,
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

    prefix_exact_match = SampleLevelMetric(
        metric_name="pem",
        sample_level_fn=ExactMatchesThinking(strip_strings=True, type_exact_match="prefix").compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

    prefix_quasi_exact_match = SampleLevelMetric(
        metric_name="pqem",
        sample_level_fn=ExactMatchesThinking(
            normalize_gold=helm_normalizer,
            normalize_pred=helm_normalizer,
            type_exact_match="prefix",
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

