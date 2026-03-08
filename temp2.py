def _redraw(self):
    self._bloch.clear()
    for state in self._active_states:
        self._bloch.add_states(state)
        self._bloch.render()
        self._bloch.axes.view_init(
        elev=self._bloch.view[1],
        azim=self._bloch.view[0]
        )
    self._fig.canvas.draw_idle()
