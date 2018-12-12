package loss;

public class MeanSquaredError extends AbstractLoss
{
	@Override
	public double calculate(double guessValue[], double trueValue[])
	{
		for (int i = 0; i < guessValue.length; i++)
		{
			error = error + Math.pow(guessValue[i] - trueValue[i], 2);
		}
		error = error / guessValue.length;

		return error;
	}
}
