import SentimentModels as s
print(s.predict_sentiment("you are a good person!!"))

i = 0

print("Welcome to Sentiment Analyzer!!")
print("-------------------------------------")
while i == 0:
    text = input("Submit a comment. Type quit to exit!!\n\n")

    if text == 'quit':
        break

    result = s.predict_sentiment(text)        

    if result[0] == 'pos':
        print("Positive Comment!! I am confident {}%".format(result[1] * 100) )

    if result[0] == 'neg':
        print("Negative Comment!! I am confident {}%".format(result[1] * 100) )

    print("\n\n")