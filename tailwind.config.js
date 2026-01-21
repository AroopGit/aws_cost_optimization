/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          dark: '#00008B',
          DEFAULT: '#1F75FE',
          light: '#74BBFB',
        }
      }
    },
  },
  plugins: [],
}

