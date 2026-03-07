# Premier League Match Predictor

Random Forest that tries to predict PL match results from historical data.

Try it out: https://huggingface.co/spaces/bolanosmanny/pl-match-predictor

## What is this

Trained on PL match data across a few seasons. Features are pretty simple - opponent, venue, kick-off hour, and day of the week. Model gets 67% precision on predicted wins. Venue and opponent ended up being the most important features.

## What the app does

- Pick any team and opponent, get a win probability
- Shows the model's confusion matrix on the test set
- Last 5 results for the team you pick
- Head to head record between the two teams

## Files

- app.py - Gradio app with all the charts and stats
- model.pkl - the trained model
- matches.csv - all the match data
- opponents.pkl / venues.pkl - label encodings

## Run it yourself

pip install gradio pandas scikit-learn matplotlib seaborn
python app.py

## Things I want to try

- More features like recent form, goal difference, league position
- Try XGBoost or logistic regression and compare
- Scrape newer seasons so the data stays fresh
- Predict draws too instead of just win/no win
