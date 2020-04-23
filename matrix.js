class Matrix {
    constructor(rows, cols) {
        this.data = this.createMatrix(rows, cols)
    }

    createMatrix(rows, cols) {
        let array = [];
        for (let i = 0; i < rows; i++) {
            array.push([]);
            for (let j = 0; j < cols; j++) {
                array[i][j] = Math.round(Math.random() * 100) / 100 * 2 - 1
            }
        }
        return array
    }

    map(func) {
        for (let i = 0; i < this.data.length; i++) {
            for (let j = 0; j < this.data[0].length; j++) {
                this.data[i][j] = func(this.data[i][j])
            }
        }
        return this;
    }

    print() {
        console.table(this.data);
    }

    matAdd(m) {
        if (m.data[0].length != this.data[0].length || m.data.length != this.data.length) {
            console.error('matrices are not the same size')
            return;
        }
        for (let i = 0; i < this.data.length; i++) {
            for (let j = 0; j < this.data[0].length; j++) {
                this.data[i][j] += m.data[i][j];
            }
        }
        return this;
    }

    add(c) {
        for (let i = 0; i < this.data.length; i++) {
            for (let j = 0; j < this.data[0].length; j++) {
                this.data[i][j] += c;
            }
        }
        return this;
    }

    negative() {
        for (let i = 0; i < this.data.length; i++) {
            for (let j = 0; j < this.data[0].length; j++) {
                this.data[i][j] *= -1;
            }
        }
        return this;
    }

    matSub(m) {
        this.matAdd(m.negative())
        return this
    }

    sub(c) {
        this.add(-c)
        return this
    }

    mult(c) {
        for (let i = 0; i < this.data.length; i++) {
            for (let j = 0; j < this.data[0].length; j++) {
                this.data[i][j] *= c;
            }
        }
        return this;
    }

    matMult(m) {
        if (m.data[0].length != this.data[0].length || m.data.length != this.data.length) {
            console.error('matrices are not the same size')
            return;
        }
        for (let i = 0; i < this.data.length; i++) {
            for (let j = 0; j < this.data[0].length; j++) {
                this.data[i][j] *= m.data[i][j];
            }
        }
        return this;
    }

    set(data) {
        this.data = data
    }

    relu() {
        for (let i = 0; i < this.data.length; i++) {
            for (let j = 0; j < this.data[0].length; j++) {
                this.data[i][j] = Math.max(0, this.data[i][j]);
            }
        }
        return this;
    }

    sig() {
        for (let i = 0; i < this.data.length; i++) {
            for (let j = 0; j < this.data[0].length; j++) {
                this.data[i][j] = 1 / (1 + Math.exp(-this.data[i][j]))
            }
        }

        return this;
    }

    transpose() {
        this.set(this.data[0].map((_, iCol) => this.data.map(row => row[iCol])));
        return this;
    }

    static normalize(m) {
        let array = []
        for (let i = 0; i < m.data.length; i++) {
            array.push([])
            let total = 0;
            for (let j = 0; j < m.data[0].length; j++) {
                total += m.data[i][j];
            }
            for (let j = 0; j < m.data[0].length; j++) {
                array[i][j] = m.data[i][j] / total
            }
        }
        return Matrix.fromArray(array);
    }

    static transpose(m) {
        let array = m.data[0].map((_, iCol) => m.data.map(row => row[iCol]));
        return Matrix.fromArray(array);
    }

    static matMult(a, b) {
        if (a.data[0].length != b.data[0].length || a.data.length != b.data.length) {
            console.error('matrices are not the same size')
            return;
        }

        let array = [];

        for (let i = 0; i < a.data.length; i++) {
            array.push([])
            for (let j = 0; j < b.data[0].length; j++) {
                array[i][j] = a.data[i][j] * b.data[i][j];
            }
        }
        return Matrix.fromArray(array);
    }

    static dot(a, b) {

        if (b.data.length != a.data[0].length) {
            console.error("rijen van b moeten evengroot zijn als colommen van a");
            return;
        }

        let array = []

        for (let i = 0; i < a.data.length; i++) {
            array.push([])
            for (let j = 0; j < b.data[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < a.data[0].length; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                array[i][j] = sum;
            }
        }

        return Matrix.fromArray(array);
    }

    static sub(a, c) {

        let array = [];

        for (let i = 0; i < a.data.length; i++) {
            array.push([])
            for (let j = 0; j < a.data[0].length; j++) {
                array[i][j] = a.data[i][j] - c;
            }
        }
        return Matrix.fromArray(array);
    }

    static matSub(a, b) {
        if (a.data[0].length != b.data[0].length || a.data.length != b.data.length) {
            console.error('matrices are not the same size')
            return;
        }

        let array = [];

        for (let i = 0; i < a.data.length; i++) {
            array.push([])
            for (let j = 0; j < b.data[0].length; j++) {
                array[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return Matrix.fromArray(array);
    }

    static fromArray(array) {
        let m = new Matrix(0, 0);
        m.set(array);
        return m;
    }
}