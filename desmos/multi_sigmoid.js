/**
 * @param {number} z 
 */
function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

/**
 * @param {number} a
 */
function sigmoid_d(a) {
  return a * (1 - a);
}

/**
 * (with bias as w[0] and x[0] = 1)
 * @param {number[]} x input vector (features)
 * @param {number[]} w weight vector
 */
export function perceptron(x, w) {
    let z = 0;
    for (let i = 0; i < x.length; i++) {
        z += x[i] * w[i];
    }

    return sigmoid(z);
}

/**
 * @param {number[]} x 
 * @param {number[][]} w 
 * @returns 
 */
function forward(x, w) {
    const xi = [1, ...x];

    const a1 = perceptron(xi, w[0]);
    const a2 = perceptron(xi, w[1]);
    
    const x3 = [1, a1, a2];
    const a3 = perceptron(x3, w[2]);

    return { a1, a2, a3, xi, x3 };
}

export function predict(x, w) {
    const { a3 } = forward(x, w);

    return a3 >= 0.5 ? 1 : 0;
}

/**
 * 
 * @param {*} D 
 * @param {number[][]} w 
 * @param {*} r 
 */
export function train_epoch(D, w, r) {
    const w_next = structuredClone(w);
    let error = 0;

    for (const [x, d] of D) {
        const { xi, a1, a2, a3, x3 } = forward(x, w_next);

        // const error = d - a3;
        error = d - a3;
        const delta_error = error * sigmoid_d(a3);
        const delta_error_hidden1 = delta_error * w_next[2][1] * sigmoid_d(a1);
        const delta_error_hidden2 = delta_error * w_next[2][2] * sigmoid_d(a2);

        for (let i = 0; i < w_next[2].length; i++) {
            w_next[2][i] += r * delta_error * x3[i];
        }
        for (let i = 0; i < w_next[0].length; i++) {
            w_next[0][i] += r * delta_error_hidden1 * xi[i];
        }
        for (let i = 0; i < w_next[1].length; i++) {
            w_next[1][i] += r * delta_error_hidden2 * xi[i];
        }
    }

    return { w: w_next, error };
}

export function train(D, r, epochs) {
    let w = [arr_rand(3), arr_rand(3), arr_rand(3)];

    for (let epoch = 0; epoch < epochs; epoch++) {
        w = train_epoch(D, w, r);
    }

    return w;
}

function arr_rand(length) {
    return Array.from({ length }, () => Math.random() * 2 - 1);
}
