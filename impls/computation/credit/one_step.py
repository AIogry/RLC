"""One-step internal credit for iterative computation."""

import jax


class OneStepCredit:
    """Stop credit through warm-up states before the final L->H pair."""

    name = 'one_step'

    @staticmethod
    def prepare_final_state(state):
        return jax.lax.stop_gradient(state)
