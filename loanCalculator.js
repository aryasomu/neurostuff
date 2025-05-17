const fetch = require('node-fetch');
const WolframAlphaAPI = require('@wolfram-alpha/wolfram-alpha-api');
const readline = require('readline');

const appId = 'TVQGRP-8JQWAT5486';

const waApi = WolframAlphaAPI(appId);

function calculateMonthlyPayment(principal, annualRate, years) {
    const monthlyRate = annualRate / 100 / 12;
    const numberOfPayments = years * 12;

    const numerator = monthlyRate * Math.pow(1 + monthlyRate, numberOfPayments);
    const denominator = Math.pow(1 + monthlyRate, numberOfPayments) - 1;

    const monthlyPayment = principal * (numerator / denominator);
    return monthlyPayment;
}

function generateAmortizationSchedule(principal, annualRate, years) {
    const monthlyPayment = calculateMonthlyPayment(principal, annualRate, years);
    const monthlyRate = annualRate / 100 / 12;
    let balance = principal;
    const schedule = [];

    for (let month = 1; month <= years * 12; month++) {
        const interest = balance * monthlyRate;
        const principalPayment = monthlyPayment - interest;
        balance -= principalPayment;

        schedule.push({
            month: month,
            payment: monthlyPayment.toFixed(2),
            principalPayment: principalPayment.toFixed(2),
            interestPayment: interest.toFixed(2),
            remainingBalance: balance < 0 ? 0 : balance.toFixed(2),
        });
    }

    return schedule;
}

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

rl.question('Enter the principal amount: ', (principal) => {
    rl.question('Enter the annual interest rate (in %): ', (annualRate) => {
        rl.question('Enter the number of years: ', (years) => {
            const p = parseFloat(principal);
            const r = parseFloat(annualRate);
            const y = parseInt(years);

            const monthlyPayment = calculateMonthlyPayment(p, r, y);
            console.log(`Monthly Payment: $${monthlyPayment.toFixed(2)}`);

            const amortizationSchedule = generateAmortizationSchedule(p, r, y);
            console.log("\nAmortization Schedule:");
            console.log("Month | Payment | Principal | Interest | Remaining Balance");
            console.log("----------------------------------------------------------");
            amortizationSchedule.forEach((entry) => {
                console.log(`${entry.month.toString().padEnd(5)} | ${entry.payment.padStart(8)} | ${entry.principalPayment.padStart(10)} | ${entry.interestPayment.padStart(8)} | ${entry.remainingBalance.padStart(18)}`);
            });

            rl.close();
        });
    });
});
