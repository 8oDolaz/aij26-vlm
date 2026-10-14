# Расшифровка метрик VLM-бенчмарков (Score self-check)

В этом документе приведено детальное описание каждой метрики, которая используется для оценки VLM (SmolVLM2) на 8 официальных бенчмарках хакатона. Для каждой строки приведены реальные примеры вопросов и эталонных ответов из логов прогона (`base_model_eval`).

Итоговый балл VLM вычисляется как среднее арифметическое точности по 8 бенчмаркам:
$$\text{Score}_{\text{VLM}} = \frac{1}{8} \sum_{i=1}^{8} \text{Accuracy}_i$$

---

## Сводная таблица метрик и примеров

| Task | Metric | Value (Base) | Описание задачи | Пример вопроса | Ожидаемый ответ |
|---|---|---|---|---|---|
| **blink** | `blink_acc` | 0.3978 | **Общее визуальное восприятие:** сопоставление ключевых точек между ракурсами, оценка глубины и перекрытий (Multiple Choice). | `Which point is closer to the camera?["A is closer","B is closer"]` | `(B)` |
| **crosspoint** | `crosspoint_accuracy` | 0.1550 | **Общее пространственное ориентирование:** локализация функциональных точек взаимодействия (affordance) и частей объектов в 3D/2D. Идет в расчет официального Score. | |  |
| crosspoint | `fine_grained_grounding` | 0.0000 | **Поиск мелких деталей:** локализация мелких элементов управления и фурнитуры на одиночном кадре. | `Point to the place for hanging the metal coat hook.` | `Геометрическая маска допустимой зоны попадания, выраженная в виде закодированной картинки` |
| crosspoint | `visibility_reasoning` | 0.4636 | **Понимание видимости:** определение, перекрыта ли деталь или видна с другого ракурса (бинарный выбор A/B). | `Is the position of the red dot in image 1 occluded in image 2? A.Yes  B.No` | `B` |
| crosspoint | `correspondence_judgement` | 0.3397 | **Сопоставление маркеров:** сопоставление красной точки с одним из цветных маркеров на втором ракурсе (выбор из кандидатов). | `I am providing you with two images of the same scene from different viewpoints. A red point is marked on the first image. You are given multiple points on the second image. Which point on the second image corresponds to the point in the first image? A. Blue dot  B. Yellow dot  C. Green dot` | `B` |
| crosspoint | `correspondence_pointing` | 0.0000 | **Кросс-ракурсное указание координат:** предсказание точных непрерывных 2D-координат точки взаимодействия на втором ракурсе. | `I am providing you with two images of the same scene from different viewpoints. A red point is marked on the first image. Locate in image 2 the corresponding point on the same affordance area to the red dot in image 1. Output coordinates representing exactly the same physical point across views.` | `Геометрическая маска допустимой зоны попадания, выраженная в виде закодированной картинки` |
| crosspoint | `level_object` | 0.1336 | **Локализация объектов целиком:** нахождение центра/точки объекта крупного плана (смартфон, пульт, стул). | `Внутри 4 основных задач (вопросов выше) есть отдельное поле level, где написано object` |  |
| crosspoint | `level_part` | 0.1735 | **Локализация частей объекта:** нахождение конкретной функциональной детали предмета (ручка кружки, подлокотник, защелка). | `Внутри 4 основных задач (вопросов выше) есть отдельное поле level, где написано part` |  |
| **embspatial** | `embspatial_acc` | 0.3282 | **Общая 3D-навигация:** пространственное ориентирование в интерьерах помещений. |  |  |
| embspatial | `ai2thor_accuracy` | 0.3139 | **Навигация в AI2-THOR:** ориентация в синтетических 3D-интерьерах симулятора. | `From your viewpoint, which object among the options is the nearest within the image?["lettuce","cup","spatula","knife"]` | `0` |
| embspatial | `mp3d_accuracy` | 0.3538 | **Навигация в Matterport3D:** пространственные вопросы по фотореалистичным сканам реальных домов. |  `In the image, how do the positions of telephone and curtain interact with each other? ['The telephone is above the curtain.', 'The telephone is left of the curtain.', 'The telephone is out of the curtain.', 'The telephone is connected to the curtain.']` | `1` |
| embspatial | `scannet_accuracy` | 0.3175 | **Навигация в ScanNet:** оценка взаимного расположения мебели по плотным 3D-сканам комнат. | `Which object from the list is situated at the greatest distance from your point of view within the image?["cabinet","window","picture","mirror"]` | `1` |
| **erqa** | `erqa_acc` | 0.2825 | **Embodied Robotic QA:** предсказание динамики сцены и результата манипуляций робота по траектории. | `If the yellow robot gripper follows the yellow trajectory, what will happen? Choices: A. Robot puts the soda on the wooden steps. B. Robot moves the soda in front of the wooden steps. C. Robot moves the soda to the very top of the wooden steps. D. Robot picks up the soda can and moves it up. Please answer directly with only the letter of the correct option and nothing else.` | `A` |
| **mmmu_val** | `mmmu_acc` | 0.2788 | **Академическая логика:** университетские задачи со схемами, графиками, диаграммами и чертежами. (4 варианта ответа) | `Which of the following best explains the overall trend shown in the <image 1>? ['Migrations to areas of Central Asia for resettlement', 'The spread of pathogens across the Silk Road', 'Invasions by Mongol tribes', 'Large-scale famine due to crop failures']` | `B` |
| **mmmu_pro_standard** | `mmmu_acc` | 0.1635 | **Усложненная логика:** мультимодальные задачи повышенной сложности с текстовым контекстом и вариантами. (10 вариантов ответа) | `Which of the following best explains the overall trend shown in the <image 1>? ['Political instability leading to population decline', 'The spread of pathogens across the Silk Road', 'Development of new trade routes', 'Climate change affecting the Silk Road', 'Migrations to areas of Central Asia for resettlement', 'Technological advancements in transportation', 'Invasions by Mongol tribes', 'Large-scale famine due to crop failures', 'Economic prosperity and population growth', 'Rise of religious conflicts along the Silk Road']` | `B` |
| **mmmu_pro_vision** | `mmmu_acc` | 0.0682 | **Чистое зрение:** сложнейшие визуальные задачи, где решение выводится строго из изображения без текстовых подсказок. | `Вопрос и варианты ответа представлены на изображении` | `B` |
| **msmu** | `msmu_overall_accuracy` | 0.3130 | **Общий балл MSMU:** доля верных ответов по всем 987 тестовым сэмплам открытого VQA. Именно эта метрика идет в итоговый Score соревнования. |  |
| msmu | `msmu_average` | 0.3012 | **Макро-среднее MSMU (Macro-Average):** среднее арифметическое точности по 8 категориям MSMU (равный вес 1/8 каждой категории). |  |  |
| msmu | `msmu_scale_estimation` | 0.0386 | **Абсолютный масштаб:** оценка физических размеров объекта (длина, ширина, высота в метрах/сантиметрах). | `What is the width of the wall shelves?` | `the wall shelves measures 0.37 meters in width.` |
| msmu | `msmu_relative_position` | 0.5341 | **Взаимное расположение:** качественное сравнение позиций (выше/ниже, ближе/дальше к камере). | `Which stands higher in the image, the white oval toilet or the green curtain?` | `the white oval toilet stands higher.` |
| msmu | `msmu_absolute_distance` | 0.2250 | **Оценка расстояний:** измерение метрической дистанции между двумя объектами сцены. | `What is the distance between the tan bed and the shower curtain?` | `The distance is 1.69 m.` |
| msmu | `msmu_scale_comparison` | 0.8679 | **Сравнение размеров:** качественное определение того, какой из двух предметов больше/выше/длиннее. | `Between the right ottoman and the smooth pillow, which one is larger?` | `The size of ottoman is 0.88m x 0.83m, pillow is 43cm x 37cm, so the right ottoman is larger.` |
| msmu | `msmu_existence` | 0.0208 | **Проверка наличия:** вопрос об отсутствующем объекте; модель обязана дать отрицательный ответ. | `How tall is the cabinet?`<br>*(когда шкафа на картинке нет)* | `No cabinet in the scene.` |
| msmu | `msmu_reference_object_estimation` | 0.4421 | **Оценка по опорному объекту:** вычисление размера целевого предмета, зная точный размер соседнего опорного предмета. | `The width of the round laundry basket is 0.47 meters, can you measure the width of the white washing machines?` | `Since the laundry basket is 0.47 m, the washing machines are 1.71 meters.` |
| msmu | `msmu_grounding` | 0.0000 | **Пространственное заземление:** определение объекта по координатам `(x, y)` или выдача координат указанного объекта. | `What is the coordinate of the square black chair?` | `It is located at (0.08,0.23) in the image.` |
| msmu | `msmu_object_counting` | 0.2812 | **Подсчет объектов:** определение точного количества одинаковых предметов в сцене. | `What is the total number of chairs in the image?` | `There are 3 chairs.` |

---

## Главные выводы

1. **Модель абсолютно слепа к координатам (`crosspoint grounding = 0.0`, `pointing = 0.0`, `msmu_grounding = 0.0`):**
   SmolVLM2 не обучена выдавать пары `[x, y]` в формате JSON или нормализованных пикселях.
2. **Провал в мелких деталях (`fine_grained_grounding = 0.0`):**
   Модель видит сцену целиком, но не различает функциональные части предметов (ручка шкафа, защелка, ножка стула).
3. **Критически слабая оценка физических размеров (`scale_estimation = 0.038`, `existence = 0.02`):**
   Модель называет случайные числа и галлюцинирует, если объект отсутствует.
4. **Высокий балл `scale_comparison = 0.868` обусловлен эвристикой:**
   При вопросе "что больше, А или Б" модель называет один из объектов и получает балл по пересечению токенов.