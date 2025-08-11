import React, { FC } from 'react';
import useStore from '../../store/useStore';
import { FONT_COLORS, FONT_STYLES } from '../../store/settings';

interface FontSettingsProps {}

const FontSettings: FC<FontSettingsProps> = () => {
  const { settings, updateSettings } = useStore();

  return (
    <div className="flex flex-col gap-4">
      <div className="form-control">
        <label className="label">
          <span className="label-text">Font Size</span>
        </label>
        <input
          type="range"
          min="12"
          max="36"
          value={settings.fontSize}
          className="range range-primary"
          onChange={(e) =>
            updateSettings({ ...settings, fontSize: e.target.valueAsNumber })
          }
        />
        <div className="flex justify-between text-xs px-2">
          <span>12</span>
          <span>24</span>
          <span>36</span>
        </div>
      </div>

      <div className="form-control">
        <label className="label">
          <span className="label-text">Font Style</span>
        </label>
        <select
          className="select select-bordered w-full"
          value={settings.fontStyle}
          onChange={(e) =>
            updateSettings({
              ...settings,
              fontStyle: e.target.value as typeof settings.fontStyle,
            })
          }
        >
          {FONT_STYLES.map((style) => (
            <option key={style}>{style}</option>
          ))}
        </select>
      </div>

      <div className="form-control">
        <label className="label">
          <span className="label-text">Font Colour</span>
        </label>
        <div className="flex flex-wrap gap-2">
          {Object.entries(FONT_COLORS).map(([name, hex]) => (
            <button
              key={name}
              className={`btn btn-sm ${settings.fontColour === name ? 'btn-active' : ''}`}
              style={{ backgroundColor: hex }}
              onClick={() =>
                updateSettings({ 
                  ...settings, 
                  fontColour: name as keyof typeof FONT_COLORS 
                })
              }
            ></button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FontSettings;
