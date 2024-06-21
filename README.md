This is a basic outline for creating a model which converts marathi to english transcriptions  . The training data is in the t.txt and the model architecture is described in the transformer.py . The main.py runs the model and also saves the model and 
the dictionary as a pickle for future use. The dataset I have manually created.
_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
Some examples - 

input- सुगंधित तंबाखू घातकसुगंधित तंबाखू मोठ्या प्रमाणात अवैध पद्धतीने शहरात येत आहे.

output-  <start>  Sugandhit  tambākhū  ghātaksugandhit  tambākhū  ghātaksugandhit  tambākhū  mothyā  pramāṇāt  avaidh  paddhatine  shaharāt  yet  āhe  .  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <end>  <start>  Sugandhit 
_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

input- बारामुला : जम्मू काश्मीरच्या बारामुला जिल्ह्यात सुरक्षा दलावर पुन्हा एकदा दहशतवादी हल्ला करण्यात आला.   

output-  <start>  Bārāmulā  :  Jammū-Kāśmīracyā  bārāmulā  jilhyāt  surakṣā  dalāvara  punhā  ekadā  dahashatvādī  hallā  karaṇyāt  ālā  .  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <space>  <end>  <start> 
