import os
import numpy as np

files = os.listdir(f'./tmp')

for elevation in [0, 2, 4, 6, 8]:
    for azimuth in [0, 8, 16, 24, 32]:
        for lightening in [0, 1, 2, 3, 4]:
            animal = np.array([])
            human = np.array([])
            airplane = np.array([])
            car = np.array([])
            truck = np.array([])
            for instance in [4, 6, 7, 8, 9]:
                if len(animal) == 0:
                    animal = np.load(f'./tmp/animal_{instance}_{elevation}_{azimuth}_{lightening}.npy', allow_pickle = True)
                else:
                    animal = np.append(animal, np.load(f'./tmp/animal_{instance}_{elevation}_{azimuth}_{lightening}.npy', allow_pickle = True), axis = 0)
                    
                if len(human) == 0:
                    human = np.load(f'./tmp/human_{instance}_{elevation}_{azimuth}_{lightening}.npy', allow_pickle = True)
                else:
                    human = np.append(human, np.load(f'./tmp/human_{instance}_{elevation}_{azimuth}_{lightening}.npy', allow_pickle = True), axis = 0)
        
                if len(human) == 0:
                    airplane = np.load(f'./tmp/airplane_{instance}_{elevation}_{azimuth}_{lightening}.npy', allow_pickle = True)
                else:
                    airplane = np.append(airplane, np.load(f'./tmp/airplane_{instance}_{elevation}_{azimuth}_{lightening}.npy', allow_pickle = True), axis = 0)
        
                if len(car) == 0:
                    car = np.load(f'./tmp/car_{instance}_{elevation}_{azimuth}_{lightening}.npy', allow_pickle = True)
                else:
                    car = np.append(car, np.load(f'./tmp/car_{instance}_{elevation}_{azimuth}_{lightening}.npy', allow_pickle = True ), axis = 0)
        
                if len(truck) == 0:
                    truck = np.load(f'./tmp/truck_{instance}_{elevation}_{azimuth}_{lightening}.npy', allow_pickle = True)
                else:
                    truck = np.append(truck, np.load(f'./tmp/truck_{instance}_{elevation}_{azimuth}_{lightening}.npy', allow_pickle = True), axis = 0)

            np.random.shuffle(animal)
            np.random.shuffle(human)
            np.random.shuffle(airplane)
            np.random.shuffle(car)
            np.random.shuffle(truck)

            np.save(f"./smallnorb_split/animal_{elevation}_{azimuth}_{lightening}.npy", animal)
            np.save(f"./smallnorb_split/human_{elevation}_{azimuth}_{lightening}.npy", human)
            np.save(f"./smallnorb_split/airplane_{elevation}_{azimuth}_{lightening}.npy", airplane)
            np.save(f"./smallnorb_split/truck_{elevation}_{azimuth}_{lightening}.npy", truck)
            np.save(f"./smallnorb_split/car_{elevation}_{azimuth}_{lightening}.npy", car)
