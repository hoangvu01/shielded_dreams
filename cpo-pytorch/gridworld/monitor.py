from .dfa import DFA

"""
A module for runtime monitoring of co-safe LTL specifications.
"""


class SafetyMonitor:

    def __init__(self, ltl_formula):
        self._dfa = DFA(ltl_formula)
        self._current_formula = ltl_formula
        self.violation_count = 0

    def step(self, true_props):
        progression = self._dfa.progress_LTL(self._current_formula, true_props)
        if progression == 'True':
            # All safety specs satisfied for some reason
            self._current_formula = progression
        elif progression == 'False':
            # Violated specification
            self.violation_count += 1
            return True
        else:
            self._current_formula = progression


if __name__ == '__main__':
    # Demo code
    formula = (
        'and',
        (
            'until',
            'b',
            ('and', 'a', 'b')
        ),
        (
            'until',
            'a',
            ('next', 'c')
        )

    )
    monitor = SafetyMonitor(formula)
    print(monitor._dfa)
    monitor.step(['a'])
    print(monitor.violation_count)
    monitor.step(['b'])
    print(monitor.violation_count)
    monitor.step(['b', 'c'])
    print(monitor.violation_count)
    monitor.step(['b'])
    print(monitor.violation_count)
    monitor.step(['a', 'b'])
    print(monitor.violation_count)
