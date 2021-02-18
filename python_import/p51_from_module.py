from machine.car import drive
from machine.tv import watch

drive()
watch()

print('='*50)

# from machine import car
# from machine import tv
from machine import car, tv

car.drive()
tv.watch()

print('='*25 + 'test' + '='*25)

from machine.test.car import drive
from machine.test.tv import watch

drive()
watch()

from machine.test import car
from machine.test import tv

car.drive()
tv.watch()

from machine import test
from machine import test

test.car.drive()
test.tv.watch()