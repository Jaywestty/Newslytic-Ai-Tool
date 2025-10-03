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
The Vice Chancellor, Federal University of Technology and Environmental Sciences, Iyin Ekiti, Prof. Gbenga Aribisala, has said that the new institution will begin the admission process in September.

Aribisala said that the admission process would follow the National Universities Commission Resource Verification exercise taking place soon in the university.

The VC, who spoke in Ado Ekiti on Sunday at a reception to celebrate the 90th birthday of his mother, Deaconess Felicia Aribisala, also canvassed support from well-meaning Nigerians to the institution, saying, “A technology-based institution of this nature is capital-intensive”.

He said, “The NUC is coming for Resource Verification of all the 36 programmes that we are trying to offer. As soon as they come, by the special grace of God, we have provided those things that will be needed.
“We have provided a modern laboratory for all the programmes. We have a library now. We have classrooms fixed.

“We have offices and furniture fixed. We have all of those things. So we are very confident we are going to scale through.

“By the time we now scale through, by the special grace of God, by September this year, we are going to ask those who are interested in our university to do Change of University, and admission will begin. That is the icing. And after that, recruitment of staff will just follow”.

The VC, who said that funding of education should not be left to the government alone, said, “Universities need a lot of funding. Funding is a major challenge. You have to provide facilities and all of those things.
“So, as I speak to you, we (FUTES) do not have enough funds. That’s why we keep appealing and going to people because the government cannot do it all alone. We have been visiting some people who are public-spirited, people who like education, tertiary education.

“If we have people who want to donate buildings, we are going to name such after them; people who want to give scholarships; people who want to build hostels in such a manner that it is their own and they will take rent and all of those things.

“I think the funding is crucial because if you look at the nature of our university, University of Technology and Environmental Sciences, it is capital-intensive, it is technology-based.

“It means we need a lot of equipment. As I said, the government cannot do everything. So we need help at this time financially,” the VC said.

Aribisala disclosed that the land issue, which could have been a challenge to the university, had been resolved amicably with an agreement made with the concerned families.

“As I speak to you now, it has been resolved. The 200 hectares that have been donated to the university are very intact.

“There has been an agreement. The community and government will also pay some compensation to the families.

“So they are now at peace. The community is not trying to force the land. I think that was the kind of misconception that happened at the time,” the Vice Chancellor said.
"""

clean = clean_and_merge_article(article)
print(clean)
# %%
