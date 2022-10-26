"""This is a tool generating a sample set, which its statistics are exactly the parameters of its distribution.
   Also the maximum and minimum values can be fixed.
"""
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize, Bounds

INFINITY = [float("-inf"), float("inf")]

DIST = {
    "normal": {
        "name": "normal",
        "generator": np.random.normal,
        "parameters": {
            "mean": [float],
            "std": [float, "> 0"]
        },
        "domain": INFINITY
    }
}

SOLVER = [
    "Nelder-Mead",
    "CG",
    "TNC",
    "BFGS",
    "L-BFGS-B"
]

BOUNDARY_SOLVER = [
    "Nelder-Mead",
    "L-BFGS-B"
]

class Tamader:
    """"""
    def __init__(self, **kwargs):
        self.logger = kwargs["logger"]
        self.max_retry = kwargs.get("max_retry", 10)
        self.logger.info("Tamader set")

    @property
    def solver(self):
        return self.__solver

    @solver.setter
    def solver(self, method):
        if method not in SOLVER:
            raise ValueError(
                "Valid solvers: {}".format(SOLVER))
        self.__solver = method

    @property
    def max_retry(self):
        return self.__max_retry

    @max_retry.setter
    def max_retry(self, num_retry):
        if not isinstance(num_retry, int) or num_retry <= 0:
            raise ValueError("max_retry should be a positive integer.")
        self.__max_retry = num_retry

    def __init_boundary(self, kwargs):
        """Initilaize boundary condition
            Returns:
                boundary (list of float): [lower bound, upper bound]
        """
        boundary = kwargs["boundary"]
        distribution = kwargs["distribution"]
        try:
            lb = float(boundary[0])
            ub = float(boundary[1])
        except ValueError as err:
            raise err
        except IndexError:
            raise IndexError("Boundary should be a list with size 2.")

        if lb >= ub:
            raise ValueError(
                f"Invalid boundary condition: lower bound = {lb}, upper bound = {ub}"
            )
        lb = max(lb, distribution["domain"][0])
        ub = min(ub, distribution["domain"][1])
        return [lb, ub]

    def _preprocess_inputs(self, kwargs):
        """Handle initial inputs
            Steps:
                1. Initialize conditions
                  - sample_size
                  - ddof
                2. Select and validate distribution
                3. Initialize and validate boundary
                4. Adjust parameters and other conditions based on boundary
                5. Validate parameters again after adjustment
                6. Set solver
            Args:
                kwargs (dict): Original request.
            Returns:
                ret (dict): Includes all the conditions and parameters.

        """
        kwargs["sample_size"] = kwargs.get("sample_size", 5)
        kwargs["ddof"] = kwargs.get("ddof", 1)
        self._validate_conditions(kwargs)

        kwargs["distribution"] = DIST[kwargs["distribution"]]
        self._validate_parameters(kwargs)

        if "boundary" in kwargs:
            kwargs["boundary"] = self.__init_boundary(kwargs)


        ret = kwargs

        if "boundary" in kwargs:
            if kwargs["sample_size"] <= 2:
                raise ValueError("sample_size should be greater than 2.")

            boundary = [val for val in kwargs["boundary"] if val not in INFINITY]
            if len(boundary) == 0:
                return ret

            boundary_l1 = sum(boundary)
            boundary_l2 = sum([val**2 for val in boundary])

            x_min, x_max = kwargs["boundary"][0], kwargs["boundary"][1]

            num_x = kwargs["sample_size"]
            ret["sample_size"] = num_x - len(boundary)

            if kwargs["distribution"]["name"] == "normal":
                var, mean = kwargs["std"]**2, kwargs["mean"]
                ret["mean"] = (num_x * mean - boundary_l1) / ret["sample_size"]
                new_squared_sum = var * (num_x - 1) + (mean **2) * num_x - boundary_l2
                ret["std"] = (new_squared_sum / ret["sample_size"] - ret["mean"] ** 2) ** 0.5
                ret["ddof"] = 0

            self._validate_parameters(ret)

        solver = kwargs.get("solver", SOLVER[0])
        if solver not in SOLVER:
            self.logger.warning(f"Invalid Solver {solver}. {SOLVER[0]} has been chosen.")
            solver = SOLVER[0]
        ret["solver"] = solver

        return ret

    def _validate_conditions(self, kwargs):
        """Validate general conditions"""
        if kwargs["sample_size"] <= 0:
            raise ValueError("sample_size should be positive.")

        if kwargs["ddof"] not in [0, 1]:
            raise ValueError("Invalid ddof value.")

        if kwargs["distribution"] not in DIST:
                        raise ValueError(
                "Invalid distribution. Available distributions are: {}".format([dist for dist in DIST.keys()])
            )

    def _validate_parameters(self, kwargs):
        """Validate parameters in a given distribution"""
        def match_condition(inp, condition):
            if condition == '> 0' and inp > 0:
                return True
            elif condition == '>= 0' and inp >= 0:
                return True
            elif condition == '<= 0' and inp <= 0:
                return True
            elif condition == '< 0' and inp < 0:
                return True
            return False


        for parameter in kwargs["distribution"]["parameters"]:
            if parameter not in kwargs:
                raise ValueError(f"Parameter {parameter} is missing.")
            inp = kwargs[parameter]
            conditions = kwargs["distribution"]["parameters"][parameter]
            dtype = conditions[0]
            if not isinstance(inp, dtype):
                raise TypeError(
                    f"Parameter {parameter} should be a {dtype}. Currently, it is a {type(inp)}."
                )
            if len(conditions) == 2:
                cond = conditions[1]
                if not match_condition(inp, cond):
                    raise ValueError(
                        f"Parameter {parameter} = {inp} should be {cond}"
                    )

        return kwargs

    def _get_initial_guess(self, obj):
        """Create an initial guess based on given conditions
           This step is essential since the optimization heavily rely on initial guess
            Args:
                obj (dict): all conditions
            Returns:
                centers (np.ndarray): Initial guess generated from random sampling
                                      and clustering. size = (n,)
        """
        sample_size = obj["sample_size"]
        generator = obj["distribution"]["generator"]
        inputs = [obj[var] for var in obj["distribution"]["parameters"]]
        samples = generator(*inputs, size=(30 * sample_size, 1))

        model = KMeans(n_clusters=sample_size)
        model.fit(samples)
        centers = model.cluster_centers_.reshape(-1,)
        return centers

    def _get_target_function(self, obj):
        """Find the loss function for a distribution
            Returns:
                func: loss function of a given distribution
        """
        if obj["distribution"]["name"] == "normal":
            def func(x):
                mean = np.mean(x)
                std = np.std(x, ddof=obj["ddof"])
                loss = (mean - obj["mean"])**2 + (std - obj["std"])**2
                return loss
        
        return func

    def _get_payload(self, obj, samples):
        """Initialized the payload for optimization iterations
            Args:
                obj (dict): all conditions
                samples (np.ndarray): initial guess
            Returns:
                payload (dict):
                  - fun: target loss function
                  - x0: initial guess
                  - tol: tolerance rate
                  - method: optimization solver
                  - boundary (optional): boundary conditions
        """
        payload = {
            "fun": self._get_target_function(obj),
            "x0": samples,
            "tol": 1e-6,
            "method": obj["solver"]
        }

        if "boundary" in obj:
            if obj["solver"] not in BOUNDARY_SOLVER:
                payload["method"] = "L-BFGS-B"
                self.logger.warning("Solver L-BFGS-B is selected for boundary conditions.")

            lb, ub = obj["boundary"]
            lb = lb * 0.999 if lb != float("-inf") else lb
            ub = ub * 0.999 if ub != float("inf") else ub
            payload["bounds"] = Bounds(lb, ub)

        return payload

    def _optimize_vector(self, payload, num_iter=0):
        """Optimization iteration
            Args:
                payload (dict): needed info for optimization iteration
                num_iter (int): retry number
            Returns:
                Status (bool): Converged or not.
                current_vector (np.ndarray): a sample that fits all the conditions,
                                             size = (n,)
        """
        outcome = minimize(**payload)
        current_vector = outcome.x
        if num_iter > self.max_retry:
            return False, current_vector
        if outcome.success and outcome.fun < 1e-6:
            return True, current_vector

        payload["x0"] = current_vector
        return self._optimize_vector(payload, num_iter=num_iter+1)

    def _fit_parameters(self, samples, obj):
        """Find a feasible solution
            Args:
                samples (np.ndarray): initial guess,
                                      shape = (sample_size,)
                obj (dict): given conditions
            Return:
                ret (np.ndarray, None): a feasible solution,
                                        shape = (sample_size,)
        """
        payload = self._get_payload(obj, samples)
        success, ret = self._optimize_vector(payload)

        if not success:
            self.logger.warning("Failed to find feasible solution.")
            return

        if "boundary" in obj:
            ret = [val for val in obj["boundary"] if val not in INFINITY] + ret.tolist()
            return np.array(ret)

        return ret

    def process(self, **kwargs):
        """Provide a solution that fits all the given conditions
           If we failed to find a feasible solution, return None
            Kwargs:
                - distribution (str): Desired statistic distribution, required
                - sample_size (int): Desired sample size, including maximum
                                     and minimum values if given. default=5
                - boundary (list): Boundary conditions if any. (Optional)
                - *parameters: Parameters of the distribution.
                               Check DIST for further information.
            Returns:
                output (np.ndarray): Feasible solution. None if it can't be found.
        """
        obj = self._preprocess_inputs(kwargs)
        init_samples = self._get_initial_guess(obj)
        output = self._fit_parameters(init_samples, obj)
        return output

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
    import scripts.common as common

    config = common.get_config(test=True)
    logger = common.get_logger(__name__)
    common.set_logger_level(logger, config['logging_level'])

    agent = Tamader(logger=logger)
    outcome = agent.process(distribution="normal", mean=1.0, std=4.0, sample_size=5, boundary=[-4, float("inf")])
    if outcome is None:
        print("Failed to find a feasible solution.")
    else:
        print(
            """
            outcome = {0}
            average = {1}
            standard deviation = {2}
            """.format(outcome, outcome.mean(), outcome.std(ddof=1)))