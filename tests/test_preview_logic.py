from ovt.preview_logic import clock, cue_index_at, neighbour_start, tail_lines

S, E = [1.0, 4.0, 4.5, 9.0], [3.0, 8.0, 5.0, 10.0]


def test_cue_lookup_edges_gaps_and_nested_cues():
    assert cue_index_at(S, E, 0.5) is None                 # before the first cue
    assert cue_index_at(S, E, 1.0) == 0 and cue_index_at(S, E, 2.999) == 0
    assert cue_index_at(S, E, 3.0) is None                 # gap (end is exclusive)
    assert cue_index_at(S, E, 4.2) == 1
    assert cue_index_at(S, E, 4.7) == 2                    # later cue wins when it starts inside an earlier one
    assert cue_index_at(S, E, 6.0) == 1                    # ...and the long cue is found again after the short one ends
    assert cue_index_at(S, E, 10.0) is None and cue_index_at([], [], 1.0) is None


def test_prev_next_neighbours():
    assert neighbour_start(S, 2.0, +1) == 4.0 and neighbour_start(S, 9.5, +1) is None
    assert neighbour_start(S, 4.2, -1) == 1.0               # 0.5 s grace: not the cue we are already in
    assert neighbour_start(S, 9.2, -1) == 4.5 and neighbour_start(S, 1.2, -1) is None


def test_clock_and_tail():
    assert clock(65) == "1:05" and clock(3725) == "1:02:05" and clock(-3) == "0:00"
    assert tail_lines(["a", "b"], 8) == "a\nb"
    assert tail_lines([str(i) for i in range(12)], 3).splitlines() == ["0", "1", "2", "… (+9)"]
