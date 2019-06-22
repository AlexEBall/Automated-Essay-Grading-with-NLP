# Automated-Essay-Grading-with-NLP

#### Data Description
The file contains 28 columns:

+ essay_id: A unique identifier for each individual student essay
+ essay_set: 1-8, an id for each set of essays
+ essay: The ascii text of a student's response
+ rater1_domain1: Rater 1's domain 1 score; all essays have this
+ rater2_domain1: Rater 2's domain 1 score; all essays have this
+ rater3_domain1: Rater 3's domain 1 score; only some essays in set 8 have this.
+ domain1_score: Resolved score between the raters; all essays have this
+ rater1_domain2: Rater 1's domain 2 score; only essays in set 2 have this
+ rater2_domain2: Rater 2's domain 2 score; only essays in set 2 have this
+ domain2_score: Resolved score between the raters; only essays in set 2 have this
+ rater1_trait1 score - rater3_trait6 score: trait scores for sets 7-8

#### Anonymization in Essays

We have made an effort to remove personally identifying information from the essays using the Named Entity Recognizer (NER) from the Stanford Natural Language Processing group and a variety of other approaches. The relevant entities are identified in the text and then replaced with a string such as "@PERSON1."

The entitities identified by NER are: "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT"

Other replacements made: "MONTH" (any month name not tagged as a date by the NER), "EMAIL" (anything that looks like an e-mail address), "NUM" (word containing digits or non-alphanumeric symbols), and "CAPS" (any capitalized word that doesn't begin a sentence, except in essays where more than 20% of the characters are capitalized letters), "DR" (any word following "Dr." with or without the period, with any capitalization, that doesn't fall into any of the above), "CITY" and "STATE" (various cities and states).

Here are some hypothetical examples of replacements made:

+ "I attend Springfield School..." --> "...I attend @ORGANIZATION1"
+ "once my family took my on a trip to Springfield." --> "once my family took me on a trip to @LOCATION1"
+ "John Doe is a person, and so is Jane Doe. But if I talk about Mr. Doe, I can't tell that's the same person." --> "...@PERSON1 is a person, and so is @PERSON2. But if you talk about @PERSON3, I can't tell that's the same person."
+ "...my phone number is 555-2106" --> "...my phone number is @NUM1"

Any words appearing in the prompt or source material for the corresponding essay set were white-listed and not anonymized.