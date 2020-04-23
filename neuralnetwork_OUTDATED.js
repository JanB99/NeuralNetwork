class NeuralNetwork {
    constructor(inputLayer, hiddenlayer, outputLayer) {

        this.lr = 0.1;

        //bias matrices
        this.bias_ih = new Matrix(hiddenlayer, 1)
        this.bias_ho = new Matrix(outputLayer, 1)

        //input nodes
        this.xs = new Matrix(inputLayer, 1)

        //weights; ih = van input naar hidden; ho = van hidden naar output
        this.ih_w = new Matrix(hiddenlayer, inputLayer)
        this.ho_w = new Matrix(outputLayer, hiddenlayer)

        //product van hidden 
        this.h = Matrix.dot(this.ih_w, this.xs).matAdd(this.bias_ih)
        //activatie van hidden met sigmoid
        this.ah = this.h.sig();

        //product van output
        this.y = Matrix.dot(this.ho_w, this.ah).matAdd(this.bias_ho)
        //activatie van output met sigmoid
        this.ay = this.y.sig();
    }

    print() {
        this.xs.print()
        this.ih_w.print()
        this.ah.print()
        this.ho_w.print()
        this.ay.print()
    }

    feedForward(xs) {
        this.xs.set(xs.data);
        this.h = Matrix.dot(this.ih_w, this.xs).matAdd(this.bias_ih)
        this.ah = this.h.sig();
        this.y = Matrix.dot(this.ho_w, this.ah).matAdd(this.bias_ho)
        this.ay = this.y.sig();
    }

    train(xs, ys, n) {

        // for (let i = 0; i < this.ho_w.data.length; i++){
        //     for (let j = 0; j < this.ho_w.data[0].length; j++){
        //         let w = this.ho_w.data[i][j] 
        //         let changeZofW = this.ah.data[i][0]; 
        //         let changeYofZ = this.getZPrime(this.ay) 
        //         let changeCofY = this.getPrimeError(ys);
        //         console.log( w - 0.01 * changeZofW * changeYofZ.data[i][0] * changeCofY)
        //     }
        // }

        for (let epoch = 0; epoch < n; epoch++) {

            let rnd = Math.floor(Math.random() * xs.length)
            this.feedForward(xs[rnd])
            this.backpropagate(ys[rnd])

        }
        let total = this.getError(ys[0]) + this.getError(ys[1]) + this.getError(ys[2]) + this.getError(ys[3])
        // console.log('Error', 1 / 4 * total)

    }

    backpropagate(ys) {

        let error_o = Matrix.matSub(ys, this.ay)
        let sigPrime_o = this.getZPrime(this.ay);
        let hTransposed = Matrix.transpose(this.ah)

        // W_new += lr * E * (sig(y) * (1 - sig(y))) * H_transposed
        // B_new += lr * E * (sig(y) * (1 - sig(y)))
        let gradient_ho = Matrix.matMult(error_o, sigPrime_o)
        let delta_ho_w = Matrix.dot(gradient_ho, hTransposed).mult(this.lr)

        this.bias_ho.matAdd(gradient_ho.mult(this.lr))
        this.ho_w.matAdd(delta_ho_w);

        ///////////////////////////////////////////

        let error_h = Matrix.dot(Matrix.transpose(Matrix.normalize(this.ho_w)), error_o);
        let sigPrime_h = this.getZPrime(this.ah);
        let iTransposed = Matrix.transpose(this.xs)

        let gradient_ih = Matrix.matMult(error_h, sigPrime_h)
        let delta_ih_w = Matrix.dot(gradient_ih, iTransposed).mult(this.lr)

        this.bias_ih.matAdd(gradient_ih.mult(this.lr))
        this.ih_w.matAdd(delta_ih_w);

        

        // let error_h_prev = Matrix.dot(Matrix.transpose(Matrix.normalize(this.h_prev_w)), error_h);
        // let sigPrime_h_prev = this.getZPrime(this.ah_prev);
        // let iTransposed = Matrix.transpose(this.xs)

        // let gradient_ih_prev = Matrix.matMult(error_h_prev, sigPrime_h_prev)
        // let delta_ih_prev_w = Matrix.dot(gradient_ih_prev, iTransposed).mult(this.lr)

        // this.bias_ih.matAdd(gradient_ih_prev.mult(this.lr))
        // this.ih_w.matAdd(delta_ih_prev_w);
    }

    predict(xs) {
        let h = Matrix.dot(this.ih_w, xs).matAdd(this.bias_ih).sig()
        let y = Matrix.dot(this.ho_w, h).matAdd(this.bias_ho).sig()
        return y;
    }

    getError(ys) {
        let sum = 0;
        for (let i = 0; i < this.ay.data.length; i++) {
            sum += Math.pow(this.ay.data[i][0] - ys.data[i][0], 2)
        }
        return sum;
    }

    getZPrime(z) {
        let m = Matrix.sub(z, 1).negative()
        return Matrix.matMult(z, m);
    }
}