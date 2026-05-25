# Dev Journal - Day 4
**Date** May 19, 2026
**Project:** Decomposition

---

## What we've done today
- We wrote the first test case for valid validation and did research on how unittests shuold look like and how to patch and use magic mocks

## Key Concepts Learned
- when you patch something it temporarily replaces that reference with a fake object. 
    - ex: When Python runs import librosa in your module, it stores a reference to the real librosa.load somewhere in memory. patch temporarily replaces that reference with a fake object. When the test ends, it puts the real one back.
    - So @patch('src.preprocessing.validation.librosa.load') is saying: "in the namespace that validation.py uses, replace librosa.load with a fake."
- magic mock:
    - Two things you can do with it:
        - Configure it — tell it what to return: mock.return_value = (fake_audio, 44100)
        - Assert on it — check it was called: mock.assert_called_once_with(some_path) this allows us to create fake objects 
- side_effect is used to simulate crashes while return_value simulated successful returns


## Action Items for Next Session
- finish the rest of the unittests for preprocessing probably gonna manually write a few more then use ai but telling it exactly what to test for efficiency