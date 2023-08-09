# Description
Many advertising platforms forbid to add contact information to the ad text but rather ask to publish it in a dedicated 
contact field only. There are several reasons for that:

1. **Consistency and Control**: By providing a dedicated contact field, the platform or website hosting the ad can ensure 
uniformity and control over how contact information is displayed. This consistency helps maintain the platform's 
branding and user interface enabling users to click or tap on the information to initiate communication.This streamlined 
process encourages engagement with the ad and increases the likelihood of conversions.

2. **Privacy and Security**: Placing contact information in a dedicated field can protect the privacy and security of 
both the advertiser and the users responding to the ad. It prevents direct exposure of personal details to potentially 
malicious actors.

3. **Verification and Filtering**: Having a separate contact field allows the platform to validate and filter contact 
information more effectively. It helps identify spam or inappropriate content, ensuring a higher quality of ads for users.

4. **Ad Policy Compliance**: Many advertising platforms have specific policies regarding advertiser-client interaction. 
These may include paying a certain comission for the mediation services of the platform. Requiring the use of a 
dedicated contact field helps ensure that advertisers adhere to these policies and not bypass the rules contacting users 
directly.

The goal of this solution is 
1. to detect if there is contact information in an ad,
2. to locate contact information in the text.

## Dataset
There are two datasets given: `train.csv` for training and `test.csv` for validation. 
Both the training and the validation datasets contain the following columns:
* `title`,
* `description`,
* `subcategory`,
* `category`,
* `price`,
* `region`,
* `city`,
* `datetime_submitted`.

Target: `is_bad`. For part II there is no target information.

Some labels in datasets may be incorrect.

For the currect solution I only use text information from the columns `title` and `description` and use `category` field 
to select appropriate model.

### Loading the Dataset
Datasets `train.csv` and `test.csv` can be downloaded by the script `./data/get_data.sh` to the `./data` folder or by 
the links:
[train](https://drive.google.com/file/d/1xGyiefcd_LtDUOWUzVWq6BRkEXugmlsC/view?usp=sharing), 
[test](https://drive.google.com/file/d/1rEwfbIlwKAPlAzIHP5oJHgo4bbRLi996/view?usp=sharing)

Mind that the datasets above are in Russian but the language model I used is suitable for both Russian and English.

# Technical limitations
The model is deployed in docker container.

Container resources:
* 4 Gb RAM
* 2 CPU kernels
* 1 GPU, 2 Gb disk memory

Execution time limits:
* 60 000 objects have to be processed no longer than 180 minutes for a presaved model on CPU and 30 minutes on GPU.
# Part 1. Predicting contact information
The model in part 1 should predict the probability that an ad contains contact information.

## 1.2 Data preprocessing

I perform data cleaning, removing extra spaces and line breaks and replacing separator characters like '*', '=', '_' 
with single spaces to reduce noise.
I concatenate title and description in one text for the language model to process. 

## 1.3 Model training

The model should predict the probability of contact information presence in an ad text inherently making it a 
classification problem.
I have selected BERT as predicting model because its bidirectional nature enables it to understand the context and 
meaning of words in a sentence, 
leading to more accurate predictions.
Also, BERT can handle variable-length texts without the need for fixed input sizes or padding.

BERT it is a large and complex model with many parameters, therefore due to the limited resources I chose smaller and 
more efficient DistilBERT - 
[rubert-tiny](https://huggingface.co/cointegrated/rubert-tiny) for Russian and English languages.


Ads in different categories can have different typical context terms and phrases. 
Also, the training and validation datasets are imbalanced and we need to perform well in each of the categories. 
Due to these reasons I train 10 BERT models for each ad category and one general model for unknown (new) categories.

Models' weights are stored in the cloud and are downloaded by the script `./data/get_model_weights.sh` during image build.

The result is a `pd.DataFrame` with columns:
* `index`: `int`,
* `prediction`: `float` in the range [0, 1].

Example:

|index  |prediction|
|-------|----------|
|0|0.12|
|1|0.95|
|...|...|
|N|0.68|

# 1.4 Metrics
Quality metric for the model is ROC-AUC averaged over ads categories*. 

```text
ROC AUC for category "Work": 0.9496350364963504
ROC AUC for category "Household utilities": 0.9578388918573931
ROC AUC for category "Electronics": 0.9487490487062404
ROC AUC for category "Transport": 0.9829547874283281
ROC AUC for category "Hobby and recreation": 0.8926088376650647
ROC AUC for category "Services": 0.9263300407050797
ROC AUC for category "Real estate": 0.9866938574044969
ROC AUC for category "Pets": 0.9537414965986395
ROC AUC for category "Personal items": 0.8117125177067024
ROC AUC for category "For business": 0.9053612291598562

Mean ROC AUC : 0.9315625743728152
```

<sub>*categories names are translated to English here.</sub>

# Part 2. Localizing contact information

Now that we have a reliable predictor we want to locate contact information in the ad text.
This can be used in two different scenarios:

Scenario 1 - We want to avoid contact information in the ads.

Reasons for that are listed in the description and the very beginning of this article. 
If the number of mal-written ads is too big to be processed by human moderators we need to automate this work. When the 
platform prevents an author from publishing an ad which violate the rules, the message should explicitly say what 
exactly is wrong and which information is considered contact on this platform. 

Scenario 2 - We want to highlight contact information in the ads.

Ads with clearly displayed contact information can offer a more seamless and convenient experience for users. 
Automatically identifying and highlighting contact details reduces the effort users need to make when trying to get in 
touch with advertisers.  Automatic detection of contact details can encourage users to initiate communication, resulting 
in higher conversion rates. Again, manually reviewing and verifying ads for contact information can be too time-consuming. 

Examples:
* for description `Call +1-888-888-88-88, there is a wrong number in the ad` -> (5, 20),
* for description `Call +1-888aaaaa888aaaa88a88, wrong number in the ad` -> (5, 27),
* for description `my ig: @nicki_i I don't answer calls`: (7, 14),
* for description `my IG: nicki_i I don't answer calls`: (7, 13),
* if there is no contact info in description -> (None, None)
* if there is more than 1 contact in description -> (None, None): `Call 19990000000 or 19991111111`.

## 1.2 Data preprocessing

I use the same data preprocessing as in the previous task.
Since we need to find contact information only in `description` I don't do the concatenation with `title`

## 1.3 Localization algorithm 

1. get contact probability from part 1 prediction;
2. split text by spaces and count words. If words count exceeds 10, split text with binary search until text size is less or equal than 10 words or contact is predicted in all parts of the text;
3. search for contact in detalized text part using sliding window and classifier model from part 1;
4. if a word appears in a window in which the model predicts contact, this word is labelled as "contact";
5. search for contiguous "contact" labels in text;
6. if there is more than 1 contiguous array of "contact" labels, this text contains several contacts, return None. Else the text contains a contact, return indices of contiguos "contact" labels.

The result is a `pd.DataFrame` with columns:
* `index`: `int`,
* `start`: `int` or `None`, beginning of the contact information;
* `finish`: `int` or `None`, end of the contact information.\
(`start` < `finish`)
  
Example output:

|index  |start|finish|
|-------|----------|-----|
|0|None|None|
|1|0|23
|2|31|123
|...|...|
|N|None|None

# Running the solution

Inference can be generated by running the script `./lib/run.py`.
To run the solution in container run the command `docker-compose -f docker-compose.yaml up`.
(`docker` and `docker-compose` must be installed)

Code is mapped to the `/app` folder in the container, `./data` is mapped to the `/data` folder in the container.
