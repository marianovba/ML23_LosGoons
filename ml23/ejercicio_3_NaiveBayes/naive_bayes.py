import numpy as np

class NaiveBayes():
    def __init__(self, alpha=1) -> None:
        self.alpha = 1e-10 if alpha < 1e-10 else alpha

    def fit(self, X, y):
        import numpy as np
        palabras = np.array()
        # TODO: Calcula la probabilidad de que una muestra sea positiva P(y=1)
        #self.prior_positives = P(y > 1|X) = (P(X|y)*P(y))/P(X)
        #self.prior_positives = P(X|y)*P(y))/P(X) 
        #self.prior_positives = P(X n y)/P(X)*P(y))/P(X)
        self.prior_positives = (((X*y)/self*self)/((X/self)*(y/self)))/(X/self)
        #self.prior_negative = self(y > 1|X) = (self(X|y)*self(y))/self(X)

        # TODO: Calcula la probabilidad de que una muestra sea negativa P(y=0)
        # self.prior_negative = P(y < 1|X) = (P(X|y)*P(y))/P(X)
        self.prior_negatives = (self*((X*y)/self*self)/(self*(X/self)*(y/self)))/(X/self)
        #self.prior_negative = self(y < 1|X) = (self(X|y)*self(y))/self(X)

        # TODO: Para cada palabra del vocabulario x_i
        # calcula la probabilidad de: P(x_i| y=1)
        # Guardalas en un arreglo de numpy:
        
        for word in palabras:
            #self.prior_negative = self(y > 1|X) = (self(X|y)*self(y))/self(X)
            #self.prior_negative = self(y < 1|X) = (self(X|y)*self(y))/self(X)
            np.append(palabras, word)
            
            
        
        
        #self._likelihoods_positives = [P(x_1| y=1), P(x_2| y=1), ..., P(x_n| y=1)]
        #self._likelihoods_positives = 
        
        # TODO:  Para cada palabra del vocabulario x_i, calcula P(x_i| y=0)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_negatives = [P(x_1| y=0), P(x_2| y=0), ..., P(x_n| y=0)]

        # self._likelihoods_negatives = _likelihoods_negatives
        return self

    def predict(self, X):
        # TODO: Calcula la distribución posterior para la clase 1 dado los nuevos puntos X
        # utilizando el prior y los likelihoods calculados anteriormente
        muestras = X.shape[0]
        predictions = np.zeros(muestras)
        
        for i in range(muestras):
            posterior_positive = self.prior_positives
            posterior_negative = self.prior_negatives
            for j in range (X.shape[1]):
                if X[i,j] == 1:
                    posterior_positive = posterior_positive * self._likelihoods_positives[j]
                    posterior_negative = posterior_negative * self._likelihoods_negatives[j]
                else:
                    posterior_positive = posterior_positive * (1-self._likelihoods_positives[j])
                    posterior_negative = posterior_negative * (1-self._likelihoods_negatives[j]) 
            

        # TODO: Determina a que clase pertenece la muestra X dado las distribuciones posteriores
            if(posterior_positive>posterior_negative):
                predictions[i] = 1
            else:
                predictions[i] = 0 
            
        return predictions
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)