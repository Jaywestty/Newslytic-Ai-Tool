#%%
#Import required library
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#Model Setup
MODEL_NAME = "Jayywestty/bart-summarizer-epoch2"

try:
    # First try: local cache only
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, local_files_only=True)
    print("✅ Loaded summarizer from local cache")
except:
    # Fallback: download from HF
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print("⬇️ Downloaded summarizer from Hugging Face")

# %%
# Put model in eval mode (best practice for inference)
model.eval()

# %%
import re
def clean_and_merge_article(article):
    # Step 1: Clean article text
    article = re.sub(r"\s+", " ", article.strip())  # collapse spaces & newlines
    article = article.replace(" ,", ",").replace(" .", ".")  # fix space before punctuation

    # Step 2: Summarize using your model
    inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
    summary_ids = model.generate(**inputs, max_length=200, min_length=80, length_penalty=2.0, num_beams=4)
    raw_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Step 3: Merge summary into one sentence
    summary = re.sub(r'\s+', ' ', raw_summary.strip())
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', summary)
    sentences = [s.strip(" .") for s in sentences if s.strip()]

    if not sentences:
        return ""
    if len(sentences) == 1:
        return sentences[0] + "."

    merged = ", ".join(sentences[:-1]) + " and " + sentences[-1]
    return merged.strip() + "."

article = """
Nigeria’s Super Eagles have arrived safely in Uyo, Akwa Ibom State, following an unexpected delay to their flight from South Africa caused by a technical fault, PUNCH Online reports.

The team, who are preparing for their decisive 2026 FIFA World Cup qualifier against Benin Republic on Tuesday, touched down at the Victor Attah International Airport, Uyo, at 8:05 am on Sunday, according to team media officer Promise Efoghe.

“Finally, Super Eagles arrive in Uyo, Akwa Ibom. The team touched down at 8.05 a.m. Sunday morning,” Efoghe confirmed in a statement issued to the media.

A video later released by the team’s media department showed players and officials disembarking from the ValueJet aircraft, signalling the end of a tense and delayed journey that had begun in Polokwane, South Africa.The Super Eagles had departed Polokwane late on Saturday, shortly after their match preparations in South Africa. However, what was meant to be a routine journey turned anxious when the ValueJet aircraft, which had earlier stopped in Luanda, Angola, for refuelling, developed a technical fault mid-air.

About 25 minutes after take-off, the pilot made an emergency U-turn back to Luanda after a loud crack appeared on the aircraft’s windscreen.

The Nigeria Football Federation confirmed in an official statement that the cracked windscreen forced the flight to return to Luanda, where all players, officials, and accompanying government delegates safely disembarked.The federation noted that the incident occurred after a routine refuelling stop, with the pilot “guiding the airplane safely back to the airport in Luanda”.

Following the incident, the NFF disclosed that ValueJet Airline worked closely with relevant Nigerian government authorities, including the Ministers of Aviation and Foreign Affairs, and the Chief of Staff to the President, to secure flight permits for a replacement aircraft to continue the journey to Nigeria.

“The ValueJet Airline and the relevant Federal Government of Nigeria authorities are working assiduously to get the necessary overflying and landing permits for another aircraft to fly from Lagos, pick the delegation in Luanda, and fly them to Uyo,” the statement read.

The replacement aircraft eventually completed the journey on Sunday morning, ending nearly 12 hours of travel disruption for the team.

Eric Chelle’s men will now turn their attention to the all-important World Cup qualifier against the Benin Republic.

The match, scheduled for Tuesday at the Godswill Akpabio International Stadium, is a crucial one for Nigeria, who are aiming to seal qualification for the 2026 FIFA World Cup after a challenging qualifying campaign."""

clean = clean_and_merge_article(article)
print(clean)