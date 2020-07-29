from keras.models import load_model

if __name__ == "__main__":

	x_test= pd.read_pickle('x_test.pkl')
	LSTM_model = load_model('single_LSTM.h5')
	predicted = LSTM_model.predict(x_test,batch_size=1)
	print("The testing MSE with vanilla LSTM is"+str(mean_squared_error(predicted,Y_test)))