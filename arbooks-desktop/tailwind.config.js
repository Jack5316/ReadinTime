import typography from "@tailwindcss/typography";
import daisyUI from "daisyui";

/** @type {import('tailwindcss').Config} */
export default {
  content : [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme : {
    extend : {},
  },
  plugins : [ typography, daisyUI ],
  daisyui : {
    themes : [
      "nord",
      "pastel",
      "garden",
      "emerald",
      "winter",
      "cyberpunk",
      {
        paper : {
          "primary" : "#ffffff",
          "secondary-content" : "#ede8d0", // beige because someone in particular wants it
          "secondary" : "#000000",
          "accent" : "#37cdbe",
          "neutral" : "#3d4451",
          "base-100" : "#000000",
          "info" : "#3abff8",
          "success" : "#36d399",
          "warning" : "#fbbd23",
          "error" : "#f87272",
        },
      },
    ]
  }
};
