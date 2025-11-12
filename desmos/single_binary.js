/**
 * (with bias as w[0] and x[0] = 1)
 * @param {number[]} x input vector (features)
 * @param {number[]} w weight vector
 */
export function predict(x, w) {
    let z = 0;
    for (let i = 0; i < x.length; i++) {
        z += x[i] * w[i];
    }

    return z > 0 ? 1 : 0;
}

/**
 * (with bias as w[0] and x[0] = 1)
 * @param {[number[], number][]} D [x: input vector, d: prediction][]
 * @param {number[]} w weight vector
 * @param {number} r learning rate
 */
export function train_epoch(D, w, r) {
    const w_next = w.slice();
    const length = w.length;
    for (let j = 0; j < D.length; j++) {
        const [x, d] = D[j];
        const y = predict(x, w);
        const error = d - y;

        for (let i = 0; i < length; i++) {
            w_next[i] += r * error * x[i];
        }
    }
    return w_next;
}

/**
 * (with bias as w[0] and x[0] = 1)
 * @param {[number[], number][]} D [x: input vector, d: prediction][]
 * @param {number[]} w weight vector
 * @param {number} epochs
 */
export function train(D, r, epochs) {
    const length = D[0][0].length;
    let w = Array(length).fill(0);

    for (let epoch = 0; epoch < epochs; epoch++) {
        w = train_epoch(D, w, r);
    }

    return w;
}
