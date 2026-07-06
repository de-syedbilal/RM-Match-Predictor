import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# setting data for the ai to predict the outcome

match_history = {
    'venue_home' : [1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
    'opp_tier' : [1, 3, 2, 1, 3, 2, 1, 2, 1, 3],
    'vini_starts' : [1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    'mbappe_starts' : [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    'bellingham_starts' : [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    'formation_type' : [1, 1, 2, 3, 1, 2, 1, 2, 3, 1],
    'weather_type' : [1, 2, 1, 3, 1, 2, 1, 1, 2, 1],
    'ref_strict' : [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    'match_outcome' : [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
}


# here we'll set the organized data using dataframe to 'df' and ai_brain part to make sure the computer takes the data from X and predicts the final results to y :)


df = pd.DataFrame(match_history)
X = df[['venue_home', 'opp_tier', 'vini_starts', 'mbappe_starts', 'bellingham_starts', 'formation_type', 'weather_type', 'ref_strict']]
y = df['match_outcome']
ai_brain = RandomForestClassifier(random_state=42)
ai_brain.fit(X,y)

# now doing the website part

st.set_page_config(page_title="Galáctico Master Tactics")
st.title('Real Madrid Match Predictor 👑')
st.write('Make sure to configure the tactical parameters below to simulate the match outcome.')
st.write('---')

# setting environment parameters here

st.subheader('📍 Match Conditions & Environment')
venue_input = st.radio('Where is the match being played?', ["Home (Santiago Bernabeu)", "Away Stadium"])
opponent_input = st.selectbox("Select Opponent Strength:", ["Elite / UCL Rival", "Mid-Table Club", "Low-Tier Club"])
weather_input = st.radio("Weather Conditions:", ["Clear Sky", "Heavy Rain", "Snowing"])

st.write("---")


# now doing the player line up checkup part
st.subheader("Starting Lineup Decisions 👟")
vini_input = st.checkbox("Is Vinícius Jr. starting?", value=True)
mbappe_input = st.checkbox("Is Kylian Mbappé starting?", value=True)
bellingham_input = st.checkbox("Is Jude Bellingham starting?", value=True)
st.write('---')


# now advanced match tactics
st.subheader("⚙️ Managerial Tactics & Officials")
formation_input = st.selectbox("Choose Team Formation:", ["4-3-3 Attacking", "4-4-2 Diamond", "5-3-2 Park the Bus"])
ref_input = st.radio("Match Referee Strictness:", ["Lenient (Allows physical play)", "Strict (Card Happy)"])


# now conversions :)

venue_number = 1 if venue_input == "Home (Santiago Bernabeu)" else 0
opp_number = 1 if opponent_input == "Elite / UCL Rival" else (2 if opponent_input == "Mid-Table Club" else 3)
vini_number = 1 if vini_input else 0
mbappe_number = 1 if mbappe_input else 0
bellingham_number = 1 if bellingham_input else 0

# formation translation

if formation_input == "4-3-3 Attacking":
    formation_number = 1
elif formation_input == "4-4-2 Diamond":
    formation_number = 2
else:
    formation_number = 3

# weather translation too

if weather_input == 'Clear Sky' :
    weather_number = 1
elif weather_input == 'Heavy Rain' :
    weather_number = 2
else:
    weather_number = 3

# referee translation 

ref_number = 1 if ref_input == "Strict (Card Happy)" else 0

# now we'll run live prediction 

st.write('---')
visitor_match_row = pd.DataFrame([{
        'venue_home': venue_number,
        'opp_tier': opp_number,
        'vini_starts': vini_number,
        'mbappe_starts': mbappe_number,
        'bellingham_starts': bellingham_number,
        'formation_type': formation_number,
        'weather_type': weather_number,
        'ref_strict': ref_number
    }])

# Create a button for the user to trigger the prediction
if st.button("Generate Tactical Prediction"):
    
    # now passing to ai to predict :0
    
    probabilities = ai_brain.predict_proba(visitor_match_row)
    win_percentage = probabilities[0][1] * 100

    st.subheader('AI Tactical Analysis Report 🤖')
    if win_percentage >= 70 :
        st.success(f"🔮 PREDICTION: HIGH PROBABILITY WIN ({win_percentage:.1f}%)")
        st.write("The AI model predicts a comfortable victory based on this tactical setup.")

    elif win_percentage >= 45 : 
        st.warning(f"🔮 PREDICTION: TIGHT CONTEST / DRAW RISK ({win_percentage:.1f}%)")
        st.write("Historical patterns flag a heavily contested match with a dangerous risk of dropping points.")

    else:
        st.error(f"🔮 PREDICTION: HIGH RISK MATCH ({win_percentage:.1f}%)")
        st.write("Warning: The AI indicates a major risk of a draw or loss under these exact conditions.")