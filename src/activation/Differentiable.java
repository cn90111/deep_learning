package activation;

public interface Differentiable
{
	public abstract double[] toDifferentiate(double[] weight);
}
