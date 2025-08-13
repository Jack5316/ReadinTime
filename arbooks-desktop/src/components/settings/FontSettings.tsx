import React, { FC } from 'react';
import useStore from '../../store/useStore';
import { FONT_COLORS } from '../../store/settings';

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

      {/* Font style selection removed: app uses a single default reading font */}

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
