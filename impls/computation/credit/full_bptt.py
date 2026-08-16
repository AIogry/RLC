"""Full internal credit for iterative computation."""


class FullBPTTCredit:
    """Keep warm-up states connected to the final loss."""

    name = 'full_bptt'

    @staticmethod
    def prepare_final_state(state):
        return state
