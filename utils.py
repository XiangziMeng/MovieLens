#coding: utf-8
import numpy as np

def to_one_hot(number, whole_size):
    temp = np.zeros(whole_size)
    temp[number - 1] = 1
    return temp

def get_data_generator(filename, batch_size):
    m_movies, n_users = 3952, 6040
    data = []
    for line in open(filename):
        user, movie, rating, timestamp = line.strip().split("::")
        user, movie = int(user), int(movie)
        data.append([movie, user])
    data = np.array(data)
    np.random.shuffle(data)
    r_batches = len(data) // batch_size
    print ("How many movies: %d" % m_movies)
    print ("How many users: %d" % n_users)
    print ("How many records for training: %d" % len(data))
    k = 0
    while True:
        k = k % r_batches 
        x_train = np.array([to_one_hot(record[0], m_movies) for record in data[k * batch_size: (k + 1) * batch_size]])
        y_train = np.array([to_one_hot(record[1], n_users) for record in data[k * batch_size: (k + 1) * batch_size]])
        yield x_train, y_train 
        k += 1

def get_movie_stats(filename):
    index2genres = {}
    for line in open(filename, 'rb'):
        try:
            line = bytes.decode(line)
            index, movie, genres = line.strip().split("::")
            index = int(index) - 1
            index2genres[index] = genres
        except Exception as e:
            pass
    return index2genres
        

if __name__ == "__main__":
    data_generator = get_data_generator("data/ratings.dat", 10)
    for x_train, y_train in data_generator:
        print (x_train, y_train)
        break
    
