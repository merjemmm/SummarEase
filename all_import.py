
# Import necessary libraries
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
import numpy as np
import re
import torch
import textstat
import math
from collections import Counter
from rouge_score import rouge_scorer
import bert_score
import textstat
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer