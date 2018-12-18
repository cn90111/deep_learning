package loss;

public class MeanSquaredError extends AbstractLoss
{
	@Override
	public double getError(double guessValue[], double trueValue[])
	{
		error = 0;
		for (int i = 0; i < guessValue.length; i++)
		{
			error = error + Math.pow(guessValue[i] - trueValue[i], 2);
		}
		error = error / guessValue.length;

		return error;
	}

	@Override
	public double[] toDifferentiate(double[] guessValue, double[] trueValue)
	{
		double[] value = new double[guessValue.length];
		for (int i = 0; i < guessValue.length; i++)
		{
			value[i] = (-2 * trueValue[i] + 2 * guessValue[i]) / guessValue.length;
		}
		return value;
	}
}
