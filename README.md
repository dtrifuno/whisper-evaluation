# Whisper Evaluation

## Results

| model      | WER   | WER (w/o punct) | CER   | CER (w/o punct) |
| ---------- | ----- | --------------- | ----- | --------------- |
| small      | 0.595 | 0.553           | 0.234 | 0.208           |
| medium     | 0.510 | 0.431           | 0.190 | 0.162           |
| large-v2   | 0.357 | 0.309           | 0.168 | 0.150           |

## Sample Errors

| stenogram                                                       | model                                                |
| --------------------------------------------------------------- | ---------------------------------------------------- |
| професионалниот и одговорност во извршувањето на оваа функција. | за оваа функција.                                    |
| За само една година, градоначалниците                           | За само една година гранчалници                      |
| Прилеп - Тополчани, Битола                                      | Прилепто-Почене Битове                               |
| Колкав поголем доказ од тоа                                     | Колкав по голем доказ от тоа                         |
| јавни претпријатија                                             | јавни предприятия                                    |
| нашето светилиште                                               | нашето светилище                                     |
| Владата на Социјалдемократскиот Сојуз на Македонија             | Владата на Социјал-демократскиот Сојуз на Македонија |
| пет и пол километри                                             | 5,5 км                                               |
| 800 илјади евра                                                 | 800.000 евра                                         |
| Благодарам.                                                     | Редактор субтитров А.Олз Корректор А.Кулаков         |
