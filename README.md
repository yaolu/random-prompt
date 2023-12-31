# random-prompt
Code and supplementary document for paper [Prompt Optimisation with Random Sampling](https://arxiv.org/abs/2311.09569)

While we clean the codebase, you can take a look at the core implementation for generating random separators in less than 10 lines of code.


1. Random vocabulary mode
```
import random
from transformers import GPT2Tokenizer

prompt = "this is a good movie [Answer:] positive"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size

# random length for separator
separator_length = random.randint(1, 5)
random_separator_ids = random.sample(range(vocab_size), separator_length)
random_separator_text = tokenizer.decode(random_separator_ids, skip_special_tokens=True)

random_prompt = prompt.replace("[Answer:]", random_separator_text)

# evaluate on training set
# ...
```

2. Random without context mode
```
from transformers import GPT2Tokenizer, GPT2LMHeadModel

prompt = "this is a good movie [Answer:] positive"

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# random length for separator
separator_length = random.randint(1, 5)
random_separator_ids = model.generate(do_sample=True, max_new_tokens=separator_length)[0]
random_separator_text = tokenizer.decode(random_separator_ids)

random_prompt = prompt.replace("[Answer:]", random_separator_text)

# evaluate on training set
# ...
```

3. Random with context mode
```
from transformers import GPT2Tokenizer, GPT2LMHeadModel

prompt = "this is a good movie [Answer:] positive"

# follow OPRO's examples format https://arxiv.org/abs/2309.03409
context = "I like this movie <INS> positive\nI don't like this movie <INS>\n"

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# random length for separator
separator_length = random.randint(1, 5)
context_input_ids = tokenizer.encode(context, return_tensors='pt')
random_separator_ids = model.generate(context_input_ids, do_sample=True, max_new_tokens=separator_length)[0]
random_separator_text = tokenizer.decode(random_separator_ids)

random_prompt = prompt.replace("[Answer:]", random_separator_text)

# evaluate on training set
# ...
``` 



