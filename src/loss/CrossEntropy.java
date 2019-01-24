package loss;

public class CrossEntropy extends AbstractLossFunction
{
	@Override
	public double getError(double guessValue[], double trueValue[])
	{
		error = 0;
		for (int i = 0; i < guessValue.length; i++)
		{
			error = error + trueValue[i] * Math.log(guessValue[i]);
			error = error + (1 - trueValue[i]) * Math.log(1 - guessValue[i]);
		}
		error = error * (-1.0 / guessValue.length);
		return error;
	}

	@Override
	public double[] toDifferentiate(double guessValue[], double trueValue[])
	{
		double[] value = new double[guessValue.length];
		for (int i = 0; i < guessValue.length; i++)
		{
			value[i] = -1 * (trueValue[i] * (1 / guessValue[i]) + (1 - trueValue[i]) * (1 / (1 - guessValue[i])));
		}
		return value;
	}
}