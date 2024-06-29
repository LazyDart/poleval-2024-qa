from load_t5 import load_plt5

tokenizer, model = load_plt5("plt5-original-base", colab=False)

# Test for polish language
input_ids = tokenizer("""W sobotnie popołudnie, spacerując po parku, Kasia spotkała dawnego przyjaciela ze szkoły, z którym od lat nie miała kontaktu, i nagle zrozumiała, że""", return_tensors="pt").input_ids

sequence_ids = model.generate(input_ids, num_beams=4,
                                no_repeat_ngram_size=1,
                                min_length=30,
                                max_length=50,
                                early_stopping=True)

sequences = tokenizer.batch_decode(sequence_ids, skip_special_tokens=True)

print(sequences)
# Result: '規, że nie ma przyjaciół."......沪 wie o tym wszystko lepiej niż ktokolwiek浑[䀀,残 po prostu się pomyliła!' <- This is not a valid polish sentence