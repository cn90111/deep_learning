package loss;

public abstract class AbstractLoss
{
	protected double error;

	public abstract double getError(double guessValue[], double trueValue[]);

	public abstract double[] toDifferentiate(double guessValue[], double trueValue[]);
}
