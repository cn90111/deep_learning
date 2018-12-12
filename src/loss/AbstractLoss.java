package loss;

public abstract class AbstractLoss
{
	protected double error = 0;

	public abstract double calculate(double guessValue[], double trueValue[]);
}
