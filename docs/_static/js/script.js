$(() => {
    /* Use wider container for the page content */
    $(".wy-nav-content").each(function () {
        this.style.setProperty("max-width", "none", "important");
    });

    /* List each class property item on a new line
       https://github.com/lightgbm-org/LightGBM/issues/5073 */
    if (window.location.pathname.toLocaleLowerCase().indexOf("pythonapi") !== -1) {
        $(".py.property").each(function () {
            this.style.setProperty("display", "inline", "important");
        });
    }

    /* Collapse specified sections in the installation guide */
    if (window.location.pathname.toLocaleLowerCase().indexOf("installation-guide") !== -1) {
        $(
            '<style>.closed, .opened {cursor: pointer;} .closed:before, .opened:before {font-family: FontAwesome; display: inline-block; padding-right: 6px;} .closed:before {content: "\\f054";} .opened:before {content: "\\f078";}</style>',
        ).appendTo("body");
        const collapsible = [
            "#build-threadless-version-not-recommended",
            "#build-mpi-version",
            "#build-gpu-version",
            "#build-cuda-version",
            "#build-rocm-version",
            "#build-java-wrapper",
            "#build-python-package",
            "#build-r-package",
            "#build-c-unit-tests",
        ];
        $.each(collapsible, (_, val) => {
            const header = `${val} > :header:first`;
            const content = `${val} :not(:header:first)`;
            $(header).addClass("closed");
            $(content).hide();
            $(header).click(() => {
                $(header).toggleClass("closed opened");
                $(content).slideToggle(0);
            });
        });
        /* Uncollapse parent sections when nested section is specified in the URL or before navigate to it from navbar */
        function uncollapse(section) {
            section.parents().each((_, val) => {
                $(val).children(".closed").click();
            });
        }
        uncollapse($(window.location.hash));
        $(".wy-menu.wy-menu-vertical li a.reference.internal").click(function () {
            uncollapse($($(this).attr("href")));
        });
    }
});
