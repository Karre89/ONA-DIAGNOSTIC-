/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // ONA Health Brand Colors
        ona: {
          primary: '#00CED1',      // Cyan
          secondary: '#20B2AA',    // Light Sea Green
          accent: '#00FFFF',       // Aqua (for glows)
          dark: '#0a1628',         // Navy background
          darker: '#060d18',       // Deeper navy
          card: 'rgba(15, 30, 50, 0.8)', // Glass card
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'ona-gradient': 'linear-gradient(135deg, #00CED1 0%, #20B2AA 50%, #008B8B 100%)',
        'ona-glow': 'radial-gradient(ellipse at center, rgba(0, 206, 209, 0.15) 0%, transparent 70%)',
      },
      boxShadow: {
        'glow': '0 0 20px rgba(0, 206, 209, 0.3)',
        'glow-lg': '0 0 40px rgba(0, 206, 209, 0.4)',
        'glass': '0 8px 32px rgba(0, 0, 0, 0.3)',
      },
      backdropBlur: {
        'xs': '2px',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 20px rgba(0, 206, 209, 0.3)' },
          '100%': { boxShadow: '0 0 40px rgba(0, 206, 209, 0.6)' },
        },
      },
    },
  },
  plugins: [],
}
