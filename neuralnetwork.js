class NeuralNetwork {
    constructor(inputLayer, hiddenlayers, outputLayer) {
        this.lr = 0.01;
        this.xs = new Matrix(inputLayer, 1)
        this.hidden = []

        for (let i = 0; i < hiddenlayers.length; i++) {
            if (i == 0) {
                this.hidden.push({
                    bias: new Matrix(hiddenlayers[i], 1).map(x => 0),
                    weights: new Matrix(hiddenlayers[i], inputLayer),
                    activated: null
                })
            } else {
                this.hidden.push({
                    bias: new Matrix(hiddenlayers[i], 1).map(x => 0),
                    weights: new Matrix(hiddenlayers[i], hiddenlayers[i - 1]),
                    activated: null
                })
            }
        }

        this.output = {
            bias: new Matrix(outputLayer, 1).map(x => 0),
            weights: new Matrix(outputLayer, hiddenlayers[hiddenlayers.length - 1]),
            activated: null
        }
        console.log(this.output)

        this.feedForward();
    }

    print() {

        this.xs.print();
        for (let i = 0; i < this.hidden.length; i++) {
            this.hidden[i].weights.print()
            this.hidden[i].activated.print()
        }

        this.output.weights.print();
        this.output.activated.print();
    }

    feedForward() {
        for (let i = 0; i < this.hidden.length; i++) {
            let w = this.hidden[i].weights;
            let b = this.hidden[i].bias;
            if (i == 0) {
                this.hidden[i].activated = Matrix.dot(w, this.xs).matAdd(b).sig()
            } else {
                this.hidden[i].activated = Matrix.dot(w, this.hidden[i - 1].activated).matAdd(b).sig()
            }
        }
        this.output.activated = Matrix.dot(this.output.weights, this.hidden[this.hidden.length - 1].activated).matAdd(this.output.bias).sig()
        return this.output.activated;
    }

    train(xs, ys, n) {

        for (let epoch = 0; epoch < n; epoch++) {

            let rnd = Math.floor(Math.random() * xs.length)
            this.xs.set(xs[rnd].data)
            this.feedForward()
            this.backpropagate(ys[rnd])

        }
        console.log('Error 0 0:', this.getError(xs[0], ys[0]))
        // console.log('Error 1 0:', this.getError(xs[1], ys[1]))
        // console.log('Error 0 1:', this.getError(xs[2], ys[2]))
        // console.log('Error 1 1:', this.getError(xs[3], ys[3]))

    }

    backpropagate(ys) {

        let error_output = Matrix.matSub(ys, this.output.activated)
        let sigPrime_output = this.getZPrime(this.output.activated);
        let hTransposed = Matrix.transpose(this.hidden[this.hidden.length - 1].activated)

        // W_new += lr * E * (sig(y) * (1 - sig(y))) * H_transposed
        // B_new += lr * E * (sig(y) * (1 - sig(y)))
        let gradient_output = Matrix.matMult(error_output, sigPrime_output)
        let delta_output_w = Matrix.dot(gradient_output, hTransposed).mult(this.lr)

        this.output.bias.matAdd(gradient_output.mult(this.lr))
        this.output.weights.matAdd(delta_output_w);

        let error;
        for (let i = this.hidden.length - 1; i >= 0; i--) {
            if (i == this.hidden.length - 1){
                error = Matrix.dot(Matrix.transpose(Matrix.normalize(this.output.weights)), error_output);
                let sigPrime = this.getZPrime(this.hidden[i].activated);
                let iTransposed = Matrix.transpose(i == 0 ? this.xs : this.hidden[i - 1].activated)
    
                let gradient = Matrix.matMult(error, sigPrime)
                let delta_w = Matrix.dot(gradient, iTransposed).mult(this.lr)
                this.hidden[i].bias.matAdd(gradient.mult(this.lr))
                this.hidden[i].weights.matAdd(delta_w);
    
            } else {
                error = Matrix.dot(Matrix.transpose(Matrix.normalize(this.hidden[i + 1].weights)), error);
                let sigPrime = this.getZPrime(this.hidden[i].activated);
                let iTransposed = Matrix.transpose(i == 0 ? this.xs : this.hidden[i - 1].activated)
    
                let gradient = Matrix.matMult(error, sigPrime)
                let delta_w = Matrix.dot(gradient, iTransposed).mult(this.lr)
                this.hidden[i].bias.matAdd(gradient.mult(this.lr))
                this.hidden[i].weights.matAdd(delta_w);
            }
        }
    }

    predict(xs) {
        this.xs.set(xs.data)
        return this.feedForward();
    }

    getError(xs, ys) {
        let sum = 0;
        for (let i = 0; i < this.output.activated.data.length; i++) {
            sum += Math.pow(ys.data[i][0] - this.predict(xs).data[i][0], 2)
        }
        return 1 / this.output.activated.data.length * sum;
    }

    getZPrime(z) {
        let m = Matrix.sub(z, 1).negative()
        return Matrix.matMult(z, m);
    }
}