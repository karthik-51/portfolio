import { Header } from "@/components/header";
import { ContactSection } from "@/components/contact-section";
import { ProjectsSection } from "@/components/projects-section";
import { CurrentlySection } from "@/components/currently-section";
import { TimelineSection } from "@/components/timeline-section";
import { Footer } from "@/components/footer";
import { cn } from "@/lib/utils";

export default function Home() {
  return (
    <div className="relative flex  w-full items-center justify-center bg-background dark:bg-background">
      <div
        className={cn(
          "absolute inset-0",
          "[background-size:25px_25px]",
          "[background-image:radial-gradient(#008080_1px,transparent_1px)]",
          "dark:[background-image:radial-gradient(#008080_1px,transparent_1px)]"
        )}
      />
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-white [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)] dark:bg-black"></div>
      <div className="min-h-screen bg-background ">
        <div className="relative z-10">
          <div className="container mx-auto px-4 l>g:px-0 max-w-4xl">
            <Header />
            <ContactSection />
            <ProjectsSection />
            <CurrentlySection />
            <TimelineSection />
          </div>
          <Footer />
        </div>
      </div>
    </div>
  );
}