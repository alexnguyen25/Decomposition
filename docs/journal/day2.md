# Dev Journal - Day 2
**Date** May 11, 2026
**Project:** Decomposition

---

## What we've done today

I mostly set up our folder structure and tested out creating scripts to download Demucs and Librosa.

## Folder Structure
decomposition/
├── main.py
├── data/
│   ├── openmic/
│   └── test_audio/
├── src/
│   ├── preprocessing/
│   ├── separation/
│   ├── feature_extraction/
│   ├── classification/
│   └── utils/
├── models/
├── tests/
├── docs/
└── requirements.txt

## Key Concepts Learned

## Folder Structure for ML

- We are using a main.py file to run the main script for the pipeline which will remain at the root.
- We have a data folder for storing our data created and test data and we will initialize routes in our main.py for files we want to test.
- Src contains all our actual pipeline folders and each folder will contain files for each pipeline stage. 
- The models will contain our weights for our classifer model we build to analyze "other"
- tests will contain our unittests
- docs will contain documentation and our journal
- BTW I'm writing this all manually and it's an absolute pain. I'm trying to learn from our day 1 journal which claude made, but claude is making me type this out for today and I think for the future... 
- We also learned how to use the terminal to create the folder structure so things like touch, mv, mkdir.

## Action Items for Next Session
- Start preprocessing stage — write the code for format conversion, sample rate normalization, and audio validation
- Write unit tests for preprocessing