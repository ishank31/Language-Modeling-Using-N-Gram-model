# INLP Assignment - 1 : Language Modeling Using N-Gram Model

### Files included:

- tokenizer.py
- language_model.py
- generator.py
- 8 txt files for perplexity scores of each LM for train and test set.
- Report containing analysis of generation and perplexity scores with examples
- README.md

The corpus files and .npy files containing the calculated probabilities can be found [here](https://drive.google.com/drive/folders/1U1jju1mIDZTXYLUEqo1zC74ExC7pFiVs?usp=sharing)

The .npy files which store the probabilities for different models should be present in the same directory as these files. This is crucial for running the files. The .py files have the path of .npy files hardcoded.

### Instructions to run the files

- tokenizer.py\
  When run, this file inputs a text and outputs the tokenized text.\
  To run this file use <code>python3 tokenizer.py</code>\
  Example usage:

```
-> python3 tokenizer.py Your text: In 'Pride and Prejudice' by Jane Austen,
Elizabeth Bennett meets Mr Darcy at a ball hosted by her friend @charles_bingly.
They dance, but Mr Darcy finds her behaviour "tolerable, but not handsome enough
to tempt him" #rude. She later visits Pemberley, Mr Darcy's estate, where she
learns more about his character. Check out more information at
https://janeausten.co.uk.

-> [['In', "'", 'Pride', 'and', 'Prejudice', "'", 'by', 'Jane', 'Austen', ',', 'Elizabeth', 'Bennett', 'meets', 'Mr', 'Darcy', 'at', 'a', 'ball', 'hosted', 'by', 'her', 'friend', '<MENTION>', '.'], ['They', 'dance', ',', 'but', 'Mr', 'Darcy', 'finds', 'her','behaviour', '"', 'tolerable', ',', 'but', 'not', 'handsome', 'enough', 'to', 'tempt', 'him', '"', '<HASHTAG>', '.'], ['She', 'later', 'visits', 'Pemberley', ',', 'Mr', "Darcy's", 'estate', ',', 'where', 'she', 'learns', 'more', 'about', 'his', 'character', '.'], ['Check', 'out', 'more', 'information', 'at', '<URL>', '.']]
```

- language_model.py\
  When run, this file inputs a sentence and outputs the likelihood score of the text.\
  To run this file use <code>python3 language_model.py <lm_type> <path_to_corpus></code>\
  Example usage:

```
-> python3 language_model.py i Ulysses-James_Joyce.txt
   Input sentence: Hello there what are you doing

-> probab score: 1.9699820759850983e-22
```

- generator.py\
  When run, this file inputs a sentence and outputs the top k choices for next word prediction with their probability.\
  To run this file use <code>python3 language_model.py <lm_type> <path_to_corpus> k </code>\
  Example usage:

```
-> python3 generator.py i Ulysses-James_Joyce.txt 4
   Input sentence: What are you doing

-> the 0.19387780092955073
   I 0.14182319825969222
   round 0.1390787516408624
   here 0.1390597917333057
```

### Generation

The punctuation marks were discarded after processing the text. Hence, the model will not generate punctuation marks during the sequence generation. The max_token has been set to 20. This can be changed by changing the <code>max_tokens</code> in <code>generate_sequence</code> function in the <code>class N_Gram_model</code>.

For unsmoothed model:

```
Prompt: ‘I am not where I want to be……’
Results:
n=1: 'I am not where I want to be </s>'

n=2: 'I am not where I want to be in the whole of the whole of the whole of the whole of the whole of the whole of the'

n=3: 'I am not where I want to be in London and when at last that he had been brought up for the sake of discovering them To be'

n=4: 'I am not where I want to be told why my views were directed to Longbourn instead of to yours A house in town I conclude They are'

n=5: 'I am not where I want to be told whether I ought or ought not to make our acquaintance in general understand Wickham's character They are gone off'
```

We observe that as we increase the value of n, the quality of the text generated also increases. This is because when n is large the the model asseses the context of the input and then generates the sequence. This leads to coherent and meaningful sequences as comapred to gibberish when n is low.
If a context is not seen by the model during training, then the model will directly output <code>EOS</code> tag and stop the generation.
For unseen data that is OOD data, the value of n does not improve the quality of the generated text. It performs bad consistently.

```
Prompt: 'Earth is the third planet in the solar system….'

Results:
n=1: 'Earth is the third planet in the solar system </s>'

n=2: 'Earth is the third planet in the solar system </s>'

n=3: 'Earth is the third planet in the solar system </s>'

n=4: 'Earth is the third planet in the solar system </s>'

n=5: 'Earth is the third planet in the solar system </s>'
```

For smoothed models:

```
Prompt: 'What are you doing here...'

Results: 'What are you doing here Stephen It flows purling widely flowing floating foampool flower unfurling They talk excitedly Little piece of original verse written by'


Prompt: 'King Macbeth was....'
Result: 'King Macbeth was <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> <UNK>'
```

We observe that for unseen context we output the <code>UNK</code> tag which is for an unkown token. For out of domain data, again the generation of <code>UNK</code> is very frequent leading to udesireable output sequence.
The calculated probabilites after smoothing are used for generation here.

### Perplexity scores

For generating the perplexity scores of a particular corpus, use the <code>write_perplexity</code> function of the <code>class N_Gram_model</code>. For this first load the .npy files which have the probabilties for training corpus stored. Then use the <code>write_perplexity</code> function. A txt file with the average perplexity and sentence-wise perplexity will be created. Following is the script to get the perplexity score files. 

```
model = N_Gram_model('Ulysses-James_Joyce.txt', 'g')
model.read_file()
model.preprocess()
model.set_up()
model.load('Ulysses_prob_dict_good_turing.npy')
model.write_perplexity()
```
