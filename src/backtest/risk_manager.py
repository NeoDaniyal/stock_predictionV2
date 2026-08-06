from dataclasses import dataclass
import pandas as pd


@dataclass
class RiskConfig:
    sizing_method: str = "equal"  # Options: 'equal', 'confidence'
    max_position_size: float = 0.35
    cash_buffer: float = 0.05


class RiskManager:

    def __init__(self, config: RiskConfig = RiskConfig()):
        self.config = config

    def calculate_weights(
        self, candidates: pd.DataFrame, prices_df: pd.DataFrame = None
    ) -> dict:
        """Calculates portfolio allocation weights for candidates based on configured method."""
        if candidates.empty:
            return {}

        n_candidates = len(candidates)

        if self.config.sizing_method == "equal":
            raw_weight = 1.0 / n_candidates
            weights = {
                ticker: min(raw_weight, self.config.max_position_size)
                for ticker in candidates["Ticker"]
            }

        elif self.config.sizing_method == "confidence":
            prob_sum = candidates["P_BUY"].sum()
            if prob_sum == 0:
                raw_weights = {
                    ticker: 1.0 / n_candidates for ticker in candidates["Ticker"]
                }
            else:
                raw_weights = {
                    row["Ticker"]: row["P_BUY"] / prob_sum
                    for _, row in candidates.iterrows()
                }

            # Cap individual position weights at max_position_size
            weights = {
                t: min(w, self.config.max_position_size) for t, w in raw_weights.items()
            }

        else:
            raise ValueError(
                f"Unknown sizing_method: {self.config.sizing_method}"
            )

        # Ensure total weights do not exceed (1 - cash_buffer)
        max_total = 1.0 - self.config.cash_buffer
        total_weight = sum(weights.values())

        if total_weight > max_total:
            scale_factor = max_total / total_weight
            weights = {t: w * scale_factor for t, w in weights.items()}

        return weights