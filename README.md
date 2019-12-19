# FGE reproduction project
Учебный проект по воспроизведению результатов статьи Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs (https://arxiv.org/pdf/1802.10026.pdf)

# Инструкция по воспроизведению результатов
Чтобы обучить изначальные модели (40 эпох и чекпоинты для FGE), требуется просто запустить соответствующие скрипты ...model training (dataset) где model это либо vgg16, либо resnet, а dataset - либо CIFAR10, либо CIFAR100. В результате их работы, в директорию со скриптами сохранятся модели и чекпоинты, которые в дальнейшем можно загрузить с помощью следующих двух строчек:
```
vgg16_cifar10 = models.vgg16_bn(pretrained=False, **{'num_classes' : 10})
vgg16_cifar10.load_state_dict(torch.load('vgg16_cifar10_32ep_1.pt'))
```
(пример для чекпоинта vgg16 на CIFAR10)

Для обучения FGE требуется запустить соответствующий скрипт ...FGE... из директории, в которой лежат соответствующие чекпоинты ```...32ep_<n_model>.pt```. Перед этим требуется в той же директории создать пустые директории
```
<model>_cifar<n_classes>_fge_<n_model>
<model> from {vgg16, resnet}
<n_classes> from {10, 100}
``` 
(кол-во директорий и значения <n_model> зависит от того, что обучается: для vgg обучаются 3 модели с <n_model> из {1, 2, 3}, для resnet 2 модели с <n_model> из {1, 2}). В результате работы, в эти новые директории сохранятся соответствующие FGE ансамбли, т.е. просто несколько моделей с названиями base_model_i.pt где i = 0,...,n, где n - размер ансамбля. Далее эти модели можно использовать точно также, как обычные модели (пример загрузки выше). Для ансамблирования достаточно подать результаты этих моделей в софтмакс и полученные вероятнсти сложить (стандартное ансамблирование). Для удобства можно использовать функцию val из скрипта ...FGE... (пример использования можно найти в jupyter ноутбуках).
  
Проще и нагляднее всего будет просто запустить ноутбуки. Результатами их работы будет всё тоже самое, только также построятся графики. При возникновении ошибок "cuda out of memory" достаточно будет просто перезапустить ноутбук, запустив в нём первые ячейки с импортами и объявлением функций, а также ячейки, на которых могла возникнуть ошибка (при достаточном кол-ве видеопамяти на GPU такого не должно произойти).

Также в ноутбуках можно посмотреть примеры загрузки моделей и их валидации на ансамблях (FGE ноутбуки, конец каждой части с конкретным датасетом).
